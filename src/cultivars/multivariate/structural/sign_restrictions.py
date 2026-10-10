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
r"""Sign-restriction identification: a set of models, reported as a set.

Declaring only the directions of responses -- a supply shock raises output
and lowers prices -- rules out rotations without pinning one down. Any
impact matrix consistent with the innovation covariance is :math:`B = L Q`
for the Cholesky factor :math:`L` and an orthogonal :math:`Q`, and the
declaration keeps the :math:`Q` for which

.. math::

   \operatorname{sign}\!\big(e_i' \Psi_h L Q e_j\big) = s_{ij}
   \qquad \text{for every declared } (i, j, s_{ij})
   \text{ and every } h \in \mathcal{H},

a region of the orthogonal group with positive measure, so what the data
and the declaration jointly deliver is a *set* of structural models.
:class:`SignRestrictedSVAR` collects that set by drawing Haar-distributed
rotations -- the QR of a Gaussian matrix with the sign of :math:`R`'s
diagonal folded into :math:`Q`, without which the draw is not uniform --
and keeping those that satisfy every declared sign at every declared lead,
flipping a restricted column wholesale when its negative passes and never
relabelling columns to rescue a draw. :class:`NarrativeSignRestrictedSVAR`
applies the same draws to a second filter, the declared history: a named
shock had a named sign in a named period, or was the most or the
overwhelming contributor to a named variable's surprise there, each
checked through the candidate's own recovered shocks
:math:`B^{-1} u_{t^\ast}`. :class:`SignRestrictedSVARResult` holds the
accepted rotations themselves, and its quantile surfaces are labelled as
summaries of the set rather than as the impulse responses of any single
model, which no rotation traces.

Two commitments shape the surface. First, the set is the object and the
prior is named. The accepted rotations carry the Haar measure over the
admissible region -- a uniform prior over rotations that nobody declared
and that concentrates mass where the responses are flat in angle, the
Baumeister-Hamilton critique -- and every quantile is pointwise, so the
median band is not a model, the Fry-Pagan critique; the result returns
the draws so that any other functional of the set can be computed, and
the exact endpoints free of both critiques are one import away in
:mod:`~cultivars.multivariate.structural.set_identification`. Second,
the set reflects identification uncertainty only. Rotations are drawn at
the reduced-form point estimate and narrative events are checked against
its residuals, and both summaries say so; reduced-form parameter
uncertainty, and the Antolín-Díaz and Rubio-Ramírez importance weights
that propagate narrative information into it, belong to the
posterior-draw version that arrives with the sampling backend.

Layout. Both models are ``_IdentificationModel`` subclasses from
``_internals``. The sign pattern is compiled by ``_validate_sign_patterns``
and the events by ``_validate_narrative_events``, both in ``_core``; the
factor comes from ``_lower_cholesky``; the draw-and-accept loops are
``_accepted_rotations`` and ``_narrative_rotations`` in ``_core``, over the
``_haar_rotation`` sampler there, which is kept below the model layer for
the same reason the Kalman recursions are. Quantile levels pass through
``_validate_quantiles``, and the summaries render through the shared
``_SIGN_QUANTILE_NOTE``, ``_UNIT_SHOCK_NOTE`` and ``_NARRATIVE_NOTE``.

References:
    Uhlig, H. (2005). What are the effects of monetary policy on output?
    Results from an agnostic identification procedure. *Journal of
    Monetary Economics*, 52(2), 381-419.

    Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural
    vector autoregressions: Theory of identification and algorithms for
    inference. *Review of Economic Studies*, 77(2), 665-696.

    Fry, R., & Pagan, A. (2011). Sign restrictions in structural vector
    autoregressions: A critical review. *Journal of Economic Literature*,
    49(4), 938-960.

    Baumeister, C., & Hamilton, J. D. (2015). Sign restrictions, structural
    vector autoregressions, and useful prior information. *Econometrica*,
    83(5), 1963-1999.

    Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative sign
    restrictions for SVARs. *American Economic Review*, 108(10),
    2802-2829.

    Mezzadri, F. (2007). How to generate random matrices from the classical
    compact groups. *Notices of the American Mathematical Society*, 54(5),
    592-604.

Example:
    A demand shock declared to raise both variables. The accepted set is a
    set of complete models -- every impact matrix factorizes the innovation
    covariance -- and the pointwise median band is traced by none of them:
    at each lead the median comes from a different rotation.

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
    >>> signs = {"demand": {"gdp": "+", "infl": "+"}}
    >>> sset = SignRestrictedSVAR(res, signs, draws=300, seed=0).identify()
    >>> sset.n_accepted, sset.shock_names
    (300, ('demand', 'unrestricted1'))
    >>> bool(np.allclose(np.einsum("nij,nkj->nik", sset.impacts, sset.impacts), res.sigma_u))
    True
    >>> draws = sset.irf_draws(8)[:, :, 0, 0]
    >>> median = sset.irf(8)[1, :, 0, 0]
    >>> nearest = np.abs(draws - median).argmin(axis=0)
    >>> len(set(nearest.tolist())) > 1
    True
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
    _validate_quantiles,
    _validate_sign_patterns,
)
from ..._internals import _IdentificationModel, _SummaryMixin
from ...exceptions import SpecificationError

__all__ = ["SignRestrictedSVAR", "SignRestrictedSVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SignRestrictedSVARResult(_SummaryMixin):
    r"""The accepted set of a sign-restricted identification.

    Not a point estimate wearing bands. Every accepted impact matrix
    :math:`B_n = L Q_n` is a complete structural model: :math:`Q_n` is
    orthogonal, so :math:`B_n B_n' = \Sigma_u` holds for each, and every
    declared sign holds at every declared lead. The object holds all of
    them. The quantile surfaces are summaries of that set, computed
    pointwise across rotations, and the summary says what that means
    rather than letting the median band read as a model.

    Note:
        The rotations are drawn uniformly (Haar) over the orthogonal group
        and kept or discarded, so the accepted set carries the uniform
        prior over rotations whether or not anyone meant to impose one --
        the Baumeister-Hamilton critique. A pointwise quantile can come
        from a different rotation at each horizon, so no single structural
        model traces a band -- the Fry-Pagan critique. The exact endpoints
        of the set, free of both, are
        :class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVAR`'s
        business; the bands here typically cover well under half of the
        exact range.

    Attributes:
        source: The reduced-form result the rotations factorize.
        impacts: ``(n_accepted, k, k)`` accepted impact matrices.
        shock_names: One label per column; restricted shocks first, in the
            order they were declared.
        horizons: The leads at which the sign restrictions were imposed.
        restriction: The declared signs, as a sentence.
        requested: Accepted draws asked for.
        attempts: Rotations actually drawn.

    See Also:
        * :class:`SignRestrictedSVAR`, :class:`NarrativeSignRestrictedSVAR`
          -- the models whose ``identify()`` returns this record.
        * :class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVARResult`
          -- the exact bounds of the same set, for one restricted shock.

    References:
        Uhlig, H. (2005). What are the effects of monetary policy on output?
        Results from an agnostic identification procedure. *Journal of
        Monetary Economics*, 52(2), 381-419.

        Fry, R., & Pagan, A. (2011). Sign restrictions in structural vector
        autoregressions: A critical review. *Journal of Economic
        Literature*, 49(4), 938-960.

        Baumeister, C., & Hamilton, J. D. (2015). Sign restrictions,
        structural vector autoregressions, and useful prior information.
        *Econometrica*, 83(5), 1963-1999.

    Example:
        A demand shock declared to raise both variables on impact. Every
        accepted impact matrix factorizes the innovation covariance and
        satisfies the signs; the acceptance rate reads how binding they
        are:

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
        >>> signs = {"demand": {"gdp": "+", "infl": "+"}}
        >>> sset = SignRestrictedSVAR(res, signs, draws=500, seed=0).identify()
        >>> sset.shock_names, sset.horizons, sset.n_accepted, sset.attempts
        (('demand', 'unrestricted1'), (0,), 500, 671)
        >>> sset.impacts.shape, round(sset.acceptance_rate, 3)
        ((500, 2, 2), 0.745)
        >>> bool(np.allclose(np.einsum("nij,nkj->nik", sset.impacts, sset.impacts), res.sigma_u))
        True
        >>> bool(np.all(sset.impacts[:, :, 0] > 0.0))
        True
        >>> sset.irf(0)[[0, 2], 0, :, 0].round(3).tolist()
        [[0.383, 0.413], [1.102, 1.024]]
    """

    source: ClosedSystemResult = field(repr=False)
    """The closed reduced-form result the rotations factorize. Kept out of the repr."""

    impacts: npt.NDArray[np.float64] = field(repr=False)
    """The ``(n_accepted, k, k)`` accepted impact matrices. Kept out of the repr."""

    shock_names: tuple[str, ...]
    """One label per column: declared shocks first, then ``unrestricted1 ...``."""

    horizons: tuple[int, ...]
    """Leads at which the traditional signs were imposed."""

    restriction: str
    """The declaration, stated as a sentence, with the narrative caveat when it applies."""

    requested: int
    """Accepted rotations asked for."""

    attempts: int
    """Rotations drawn, accepted or not."""

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> SignRestrictedSVAR(res, {"d": {"gdp": "+"}}, draws=5).identify().names
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
            >>> SignRestrictedSVAR(res, {"d": {"y1": "+"}}, draws=5).identify().k_endog
            3
        """
        return len(self.source.names)

    @property
    def n_accepted(self) -> int:
        """Accepted rotations in the set.

        Equal to ``requested`` unless the attempt budget ran out first.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> signs = {"d": {"gdp": "+", "infl": "+"}}
            >>> full = SignRestrictedSVAR(res, signs, draws=50, seed=0).identify()
            >>> cut = SignRestrictedSVAR(res, signs, draws=50, max_attempts=20, seed=0).identify()
            >>> full.n_accepted, cut.n_accepted < 50, cut.attempts
            (50, True, 20)
        """
        return int(self.impacts.shape[0])

    @property
    def acceptance_rate(self) -> float:
        """Accepted rotations per draw, a direct read on how binding the signs are.

        Example:
            Adding a second restricted shock cuts the rate roughly in half
            here; the rate is the Haar measure of the admissible region, up
            to the column flips the sampler allows:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
            >>> one = {"demand": {"gdp": "+", "infl": "+"}}
            >>> two = {**one, "supply": {"gdp": "+", "infl": "-"}}
            >>> a = SignRestrictedSVAR(res, one, draws=200, seed=0).identify().acceptance_rate
            >>> b = SignRestrictedSVAR(res, two, draws=200, seed=0).identify().acceptance_rate
            >>> bool(0.0 < b < a <= 1.0)
            True
        """
        return self.n_accepted / self.attempts

    def irf_draws(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """Structural impulse responses for every accepted rotation.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(n_accepted, horizon + 1, k, k)``; entry
            ``[n, h, i, j]`` is variable ``i``'s response at lead ``h`` to
            one standard deviation of shock ``j`` under rotation ``n``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> sset = SignRestrictedSVAR(res, {"d": {"gdp": "+"}}, draws=20, seed=0).identify()
            >>> draws = sset.irf_draws(3)
            >>> draws.shape, bool(np.allclose(draws[:, 0], sset.impacts))
            ((20, 4, 2, 2), True)
            >>> running = sset.irf_draws(3, cumulative=True)
            >>> bool(np.allclose(running, np.cumsum(draws, axis=1)))
            True
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

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> sset = SignRestrictedSVAR(res, {"d": {"gdp": "+"}}, draws=50, seed=0).identify()
            >>> bands = sset.irf(3)
            >>> bands.shape, sset.irf(3, quantiles=(0.5,)).shape
            ((3, 4, 2, 2), (1, 4, 2, 2))
            >>> bool(np.all(bands[0] <= bands[1]) and np.all(bands[1] <= bands[2]))
            True
            >>> bool(np.all(bands[0, 0, 0, 0] > 0.0))
            True
            >>> sset.irf(3, quantiles=(0.0, 1.0))
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: quantiles must lie in (0, 1); got (0.0, 1.0).
        """
        levels = _validate_quantiles(quantiles)
        if any(not 0.0 < q < 1.0 for q in levels):
            raise SpecificationError(f"quantiles must lie in (0, 1); got {levels}.")
        draws = self.irf_draws(horizon, cumulative=cumulative)
        return np.quantile(draws, levels, axis=0)

    def fevd(
        self, horizon: int = 20, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Pointwise quantiles of the variance decomposition across the set.

        Each accepted rotation is complete, so within a rotation the shares
        sum to one; the quantile surfaces, being pointwise, need not --
        except with two variables, where the second share is one minus the
        first and the quantiles mirror exactly.

        Args:
            horizon: Largest lead to return.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), horizon + 1, k, k)``; entry
            ``[q, h, i, j]`` is the ``q``-quantile of shock ``j``'s share of
            variable ``i``'s forecast-error variance at lead ``h``.

        Raises:
            SpecificationError: If a quantile is outside the open unit
                interval.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 3)), order=1).fit()
            >>> sset = SignRestrictedSVAR(res, {"d": {"y1": "+"}}, draws=50, seed=0).identify()
            >>> shares = sset.fevd(3)
            >>> shares.shape, bool(np.all((0.0 <= shares) & (shares <= 1.0)))
            ((3, 4, 3, 3), True)
            >>> draws = sset.irf_draws(3)
            >>> explained = np.cumsum(draws**2, axis=1)
            >>> own = explained / explained.sum(axis=3, keepdims=True)
            >>> bool(np.allclose(own.sum(axis=3), 1.0))
            True
            >>> bool(np.allclose(shares[1].sum(axis=2), 1.0))
            False
        """
        levels = _validate_quantiles(quantiles)
        if any(not 0.0 < q < 1.0 for q in levels):
            raise SpecificationError(f"quantiles must lie in (0, 1); got {levels}.")
        theta = self.irf_draws(horizon)
        explained = np.cumsum(theta**2, axis=1)
        shares = explained / explained.sum(axis=3, keepdims=True)
        return np.quantile(shares, levels, axis=0)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        No parameter rows: the deliverable is a set, and a table of medians
        would misread as a model. The metadata carries the draw accounting
        and the notes carry the declaration, the unit-shock convention and
        the pointwise-quantile caveat.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> sset = SignRestrictedSVAR(res, {"d": {"gdp": "+"}}, draws=20, seed=0).identify()
            >>> table = sset._summary_table()
            >>> table.title, table.rows, len(table.notes)
            ('Sign-Restricted SVAR (identified set)', (), 3)
            >>> meta = dict(table.metadata)
            >>> meta["Accepted"], meta["Requested"], meta["Horizons"]
            ('20', '20', '0')
        """
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
    r"""Set identification by declared response signs, Uhlig (2005) via RWZ draws.

    Rotations :math:`Q` of the Cholesky factor are drawn uniformly over the
    orthogonal group (the QR of a Gaussian matrix with the sign fix that
    makes it Haar, Rubio-Ramírez, Waggoner and Zha 2010) and kept when
    every declared sign holds at every declared lead:

    .. math::

       \operatorname{sign}\!\big(e_i' \Psi_h L Q e_j\big) = s_{ij},
       \qquad h \in \mathcal{H},

    for each restricted column :math:`j` and each declared pair
    :math:`(i, s_{ij})`. A restricted column may be flipped wholesale,
    since a shock and its negative are the same rotation; what is never
    done is relabeling columns to rescue a draw, so a shock's identity is
    its declared position.

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

    Attributes:
        _source: The closed reduced-form result being identified.
        _compiled: Per-column ``(variable index, +-1.0)`` sign pairs.
        _labels: The shock labels, declared first, then ``unrestrictedN``.
        _leads: The horizons at which the signs bind.
        _draws: Accepted rotations requested.
        _budget: Rotations to attempt before giving up.
        _seed: The seed behind the draws.
        _restriction: The declaration, stated as a sentence.

    See Also:
        * :class:`SignRestrictedSVARResult` -- the accepted set returned.
        * :class:`NarrativeSignRestrictedSVAR` -- the same draws filtered
          additionally by declared history.
        * :class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVAR`
          -- the exact bounds of the set for one restricted shock, with no
          prior over rotations.

    References:
        Uhlig, H. (2005). What are the effects of monetary policy on output?
        Results from an agnostic identification procedure. *Journal of
        Monetary Economics*, 52(2), 381-419.

        Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural
        vector autoregressions: Theory of identification and algorithms for
        inference. *Review of Economic Studies*, 77(2), 665-696.

    Example:
        Two shocks declared, a demand shock moving both variables up and a
        supply shock moving them apart. The set is sharper than with one
        shock -- fewer draws pass -- and the median impact responses carry
        the declared signs:

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
        >>> signs = {"demand": {"gdp": "+", "infl": "+"}, "supply": {"gdp": "+", "infl": "-"}}
        >>> sset = SignRestrictedSVAR(res, signs, draws=200, seed=0).identify()
        >>> sset.shock_names, sset.n_accepted, sset.attempts
        (('demand', 'supply'), 200, 753)
        >>> median = sset.irf(0)[1, 0]
        >>> np.sign(median).tolist()
        [[1.0, 1.0], [1.0, -1.0]]
        >>> SignRestrictedSVAR(res, {**signs, "third": {"gdp": "+"}})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: 3 restricted shocks exceed the 2 shocks ...
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
        """Validate the source system and the full declaration.

        Args:
            result: The fitted closed reduced-form result to identify.
            restrictions: Shock label to sign pattern, variable to ``"+"``
                or ``"-"``.
            horizons: Non-empty leads at which every sign holds.
            draws: Accepted rotations to collect, at least one.
            max_attempts: Attempt budget, at least one; ``None`` for a
                thousand per draw.
            seed: Seed for the rotation draws.

        Raises:
            SpecificationError: If the result is not a closed system, the
                declaration is empty, names more shocks than the system has,
                an unknown variable or an unknown symbol, or ``horizons``,
                ``draws`` or ``max_attempts`` are out of range.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> model = SignRestrictedSVAR(res, {"demand": {"gdp": "+", "infl": "+"}}, draws=5)
            >>> model._labels, model._leads, model._compiled, model._draws, model._budget
            (('demand', 'unrestricted1'), (0,), (((0, 1.0), (1, 1.0)),), 5, 5000)
            >>> SignRestrictedSVAR(res, {"demand": {"gdp": "+"}}, draws=0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: draws must be at least 1; got 0.
            >>> SignRestrictedSVAR(res, {"demand": {"gdp": "+"}}, max_attempts=0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: max_attempts must be at least 1; got 0.
            >>> SignRestrictedSVAR(res, {"demand": {"gdp": "+"}}, horizons=(-1,))
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizons must be a non-empty collection ...
            >>> SignRestrictedSVAR(res, {})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: restrictions must declare at least one ...
        """
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

        Example:
            The same seed gives the same set; on a system whose responses
            alternate in sign, requiring both variables to rise at leads
            zero and one is nearly unsatisfiable and the budget runs out:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
            >>> signs = {"demand": {"gdp": "+", "infl": "+"}}
            >>> first = SignRestrictedSVAR(res, signs, draws=30, seed=1).identify()
            >>> again = SignRestrictedSVAR(res, signs, draws=30, seed=1).identify()
            >>> bool(np.allclose(first.impacts, again.impacts)), first.attempts == again.attempts
            (True, True)
            >>> flip = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     flip[t] = -0.9 * flip[t - 1] + B @ rng.standard_normal(2)
            >>> alternating = VAR(flip, order=1, names=("gdp", "infl")).fit()
            >>> SignRestrictedSVAR(
            ...     alternating, signs, horizons=(0, 1), draws=10, max_attempts=200
            ... ).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: no rotation satisfied the declared signs ...
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
    r"""Sign restrictions sharpened by declared history, Antolín-Díaz and Rubio-Ramírez (2018).

    Traditional signs restrict what a shock *would do*; narrative events
    restrict what the shocks *did*. Two kinds are declared, both from the
    paper. A shock-sign event states the sign of a named shock in a named
    period -- the monetary shock was contractionary in the Volcker quarter:
    :math:`\varepsilon_{j,t^\ast} = (B^{-1} u_{t^\ast})_j > 0`. A
    contribution event states that a named shock was the ``"most"``
    important contributor, or the ``"overwhelming"`` one -- larger than all
    others combined -- to the unexpected movement of a named variable in a
    named period:

    .. math::

       \lvert B_{ij}\, \varepsilon_{j,t^\ast} \rvert
       > \max_{l \neq j} \lvert B_{il}\, \varepsilon_{l,t^\ast} \rvert
       \quad\text{or}\quad
       \lvert B_{ij}\, \varepsilon_{j,t^\ast} \rvert
       > \sum_{l \neq j} \lvert B_{il}\, \varepsilon_{l,t^\ast} \rvert .

    Each accepted rotation must reproduce the declared history through its
    own recovered shocks, which is what makes the resulting set sharper
    than the traditional one: history is a filter rotations rarely pass by
    accident. Periods index the effective sample -- one row per residual
    row of the result being identified -- exactly as a proxy's instrument
    rows do. The events are checked against the point-estimate residuals,
    and the summary carries the caveat that the full posterior treatment
    belongs to the sampling backend.

    Note:
        A shock-sign event on a column with no traditional signs is
        vacuous: the sampler may flip any column, and flipping the column
        flips the recovered shock, so every draw passes. Shock-sign events
        bind only alongside traditional signs on the same shock, which fix
        the column's orientation; contribution events are flip-invariant
        and bind on their own.

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
            event is declared -- :class:`SignRestrictedSVAR` is that model --
            or any declaration is malformed.

    Attributes:
        _source: The closed reduced-form result being identified.
        _compiled: Per-column traditional sign pairs, length ``k``, empty
            where a column declares none.
        _labels: The shock labels, declared first, event-only next, then
            ``unrestrictedN``.
        _leads: The horizons at which the traditional signs bind.
        _shock_events: Per-column ``(event index, +-1.0)`` shock-sign pairs.
        _contribution_events: ``(column, variable index, event index,
            overwhelming?)`` contribution requirements.
        _periods: The unique residual rows the events reference.
        _draws: Accepted rotations requested.
        _budget: Rotations to attempt before giving up.
        _seed: The seed behind the draws.
        _restriction: The declaration, stated as a sentence.

    See Also:
        * :class:`SignRestrictedSVAR` -- the traditional signs alone.
        * :class:`SignRestrictedSVARResult` -- the accepted set returned,
          with the narrative caveat appended to its restriction sentence.
        * :class:`~cultivars.multivariate.structural.external_instruments.ProxySVAR`
          -- the other scheme that brings outside information about
          particular periods, as a series rather than as events.

    References:
        Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative sign
        restrictions for SVARs. *American Economic Review*, 108(10),
        2802-2829.

    Example:
        The period in which the true demand shock is largest relative to
        the other shock is declared as a negative demand shock that
        overwhelmingly drove output. The accepted set shrinks -- the
        acceptance rate halves -- and the 16th percentile of output's
        impact response moves from under 0.4 to near the true 1.0:

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
        >>> period = int(np.argmax(np.abs(eps[1:, 0]) - np.abs(eps[1:, 1])))
        >>> period, eps[period + 1].round(2).tolist()
        (238, [-3.9, 0.46])
        >>> signs = {"demand": {"gdp": "+", "infl": "+"}}
        >>> plain = SignRestrictedSVAR(res, signs, draws=300, seed=0).identify()
        >>> narrative = NarrativeSignRestrictedSVAR(
        ...     res,
        ...     signs,
        ...     shock_signs=(("demand", period, "-"),),
        ...     contributions=(("demand", "gdp", period, "overwhelming"),),
        ...     draws=300,
        ...     seed=0,
        ... ).identify()
        >>> narrative.shock_names, narrative.n_accepted, narrative.attempts
        (('demand', 'unrestricted1'), 300, 806)
        >>> round(plain.acceptance_rate, 2), round(narrative.acceptance_rate, 2)
        (0.75, 0.37)
        >>> plain.irf(0)[0, 0, 0, 0].round(3), narrative.irf(0)[0, 0, 0, 0].round(3)
        (np.float64(0.387), np.float64(0.948))
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
        """Validate the source system and every declaration.

        Args:
            result: The fitted closed reduced-form result to identify.
            restrictions: Traditional sign patterns, or ``None``.
            shock_signs: ``(shock label, period, sign)`` events.
            contributions: ``(shock label, variable, period, kind)`` events.
            horizons: Non-empty leads at which the traditional signs hold.
            draws: Accepted rotations to collect, at least one.
            max_attempts: Attempt budget, at least one; ``None`` for a
                thousand per draw.
            seed: Seed for the rotation draws.

        Raises:
            SpecificationError: If the result is not a closed system, no
                event is declared, the labels across signs and events exceed
                ``k``, an event names an unknown shock, variable, sign or
                kind, a period lies outside the effective sample, or
                ``horizons``, ``draws`` or ``max_attempts`` are out of
                range.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> model = NarrativeSignRestrictedSVAR(
            ...     res,
            ...     {"demand": {"gdp": "+"}},
            ...     shock_signs=(("demand", 3, "+"), ("other", 7, "-")),
            ...     contributions=(("demand", "gdp", 3, "most"),),
            ... )
            >>> model._labels, model._compiled, model._periods
            (('demand', 'other'), (((0, 1.0),), ()), (3, 7))
            >>> model._shock_events, model._contribution_events
            ((((0, 1.0),), ((1, -1.0),)), ((0, 0, 0, False),))
            >>> NarrativeSignRestrictedSVAR(res, {"demand": {"gdp": "+"}})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: declare at least one narrative event; ...
            >>> NarrativeSignRestrictedSVAR(res, shock_signs=(("demand", 120, "+"),))
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: shock-sign event on 'demand' references ...
            >>> NarrativeSignRestrictedSVAR(res, contributions=(("demand", "gdp", 3, "all"),))
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: contribution kind must be 'most' or ...
            >>> NarrativeSignRestrictedSVAR(
            ...     res, shock_signs=(("a", 1, "+"), ("b", 1, "+"), ("c", 1, "+"))
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: 3 labelled shocks exceed the 2 shocks ...
        """
        super().__init__(result)
        k = self.k_endog
        if not shock_signs and not contributions:
            raise SpecificationError(
                "declare at least one narrative event; with traditional signs "
                "alone, SignRestrictedSVAR is that model."
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

        Each candidate's structural shocks at the event periods are
        recovered as ``B^{-1} u_t``; traditional and shock-sign
        requirements on a column are checked jointly under the column
        flip, and contribution events, which are flip-invariant, are
        checked once after every column's orientation is settled.

        Returns:
            The accepted set.

        Raises:
            SpecificationError: If no rotation is accepted within the attempt
                budget -- the narrative events contradict the traditional
                signs, or the history at this reduced form.
            NumericalError: If the innovation covariance is not positive
                definite.

        Example:
            Every accepted rotation reproduces the declared history through
            its own shocks; declaring the wrong sign for a dominant event
            exhausts the budget:

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
            >>> signs = {"demand": {"gdp": "+", "infl": "+"}}
            >>> event = (("demand", "gdp", 238, "overwhelming"),)
            >>> sset = NarrativeSignRestrictedSVAR(
            ...     res, signs, shock_signs=(("demand", 238, "-"),), contributions=event, draws=100
            ... ).identify()
            >>> shocks = np.linalg.solve(sset.impacts, np.tile(res.resid[238], (100, 1))[..., None])
            >>> bool(np.all(shocks[:, 0, 0] < 0.0))
            True
            >>> shares = np.abs(sset.impacts[:, 0, :] * shocks[:, :, 0])
            >>> bool(np.all(shares[:, 0] > shares[:, 1]))
            True
            >>> NarrativeSignRestrictedSVAR(
            ...     res, signs, shock_signs=(("demand", 238, "+"),), contributions=event,
            ...     draws=10, max_attempts=500,
            ... ).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: no rotation satisfied the declared signs ...
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
