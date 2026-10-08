# filepath: /src/cultivars/multivariate/structural/zero_restrictions.py
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
r"""Zero-restriction identification: triangularity on impact or in the long run.

A reduced-form innovation covariance admits infinitely many factorizations
:math:`\Sigma_u = B B'`, one per orthogonal rotation of any of them, and
every one tells a different economic story. The schemes here choose by the
oldest device in the literature: zeros, enough of them to leave one
factorization standing. Four arrangements are offered. :class:`RecursiveSVAR`
puts a triangle on *impact* -- a declared causal ordering in which each
variable responds contemporaneously only to shocks at or before its own
position, Sims (1980), the Cholesky factor in closed form.
:class:`LongRunSVAR` puts the triangle at the *infinite horizon* -- an
ordering of permanence in which each shock has no cumulated effect on the
variables before its position, Blanchard-Quah (1989) -- as the Cholesky
factor of :math:`F \Sigma_u F'` mapped back through the long-run matrix
:math:`F = A(1)^{-1} M(1)`. :class:`ShortRunSVAR` is the general AB model,
:math:`A u_t = B \varepsilon_t` with zeros anywhere in either matrix,
estimated by maximum likelihood and tested for over-identification when
the zeros exceed :math:`k(k+1)/2 - k(k-1)/2` worth of freedom
(Amisano-Giannini 1997). :class:`MixedSVAR` takes :math:`k(k-1)/2` zeros
split between the impact matrix and the long-run matrix and solves for
the rotation of the Cholesky factor that honours all of them, Galí
(1999). Every scheme returns the same :class:`SVARResult`: impact columns,
structural impulse responses, variance shares, recovered shocks and
historical decompositions, computed per identified column.

Two commitments shape the surface. First, the restriction is an argument,
never a default. Each model constructs from a fitted closed reduced-form
result rather than from data: the reduced form supplies every estimable
quantity, and what the model adds is exactly the declared zeros, which the
result restates as a sentence in its summary -- the ordering, the cell
pattern, the horizons -- so that the identifying assumption travels with
the numbers it produced. Second, the covariance is reproduced or the
scheme refuses. A recursive or long-run factor reproduces :math:`\Sigma_u`
by construction; the long-run one is checked anyway, because a nearly
nonstationary system makes :math:`F` ill-conditioned enough to break it; a
just-identified AB pattern that cannot reproduce it is the rank condition
failing and is reported as that, not as an estimate; a mixed pattern no
rotation can satisfy is refused with the best violation. Where the zeros
over-identify, the likelihood-ratio statistic against the unrestricted
covariance is the data's verdict and is printed; where they exactly
identify, the summary says the data cannot contradict them.

Layout. All four models are ``_IdentificationModel`` subclasses from
``_internals``; orderings are validated by ``_validate_ordering`` and cell
patterns by ``_validate_impact_pattern``, both in ``_core``; the factor
comes from ``_lower_cholesky`` and the long-run matrix from
``_long_run_matrix``, which refuses a unit root; the AB likelihood is
``_ShortRunObjective`` under ``_maximize_likelihood`` and the mixed
rotation is ``_MixedHorizonObjective`` under ``_solve``, all in
``_internals``. :class:`SVARResult` renders through the shared
``_UNIT_SHOCK_NOTE`` and, when columns are missing, the
``_PARTIAL_IDENTIFICATION_NOTE``; it is also the record returned by the
partial and statistical schemes in
:mod:`~cultivars.multivariate.structural.external_instruments`,
:mod:`~cultivars.multivariate.structural.heteroskedacity` and
:mod:`~cultivars.multivariate.structural.non_gaussian`, and the point
surface :mod:`~cultivars.multivariate.structural.stochastic_volatility`
lands on. Restrictions that do not point-identify belong to
:mod:`~cultivars.multivariate.structural.sign_restrictions` and
:mod:`~cultivars.multivariate.structural.set_identification`.

References:
    Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1),
    1-48.

    Blanchard, O. J., & Quah, D. (1989). The dynamic effects of aggregate
    demand and supply disturbances. *American Economic Review*, 79(4),
    655-673.

    Amisano, G., & Giannini, C. (1997). *Topics in Structural VAR
    Econometrics* (2nd ed.). Springer.

    Galí, J. (1999). Technology, employment, and the business cycle: Do
    technology shocks explain aggregate fluctuations? *American Economic
    Review*, 89(1), 249-271.

    Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive
    Analysis*. Cambridge University Press.

Example:
    The same reduced form under the impact triangle and the long-run
    triangle. Both reproduce the innovation covariance; they disagree on
    the impact matrix, and the disagreement is the identifying assumption:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
    >>> B = (np.eye(2) - A) @ np.array([[1.0, 0.0], [0.5, 1.0]])
    >>> eps = rng.standard_normal((602, 2))
    >>> y = np.zeros((602, 2))
    >>> for t in range(1, 602):
    ...     y[t] = A @ y[t - 1] + B @ eps[t]
    >>> res = VAR(y, order=1, names=("dy", "u")).fit()
    >>> recursive = RecursiveSVAR(res).identify()
    >>> long_run = LongRunSVAR(res).identify()
    >>> recursive.impact.round(2).tolist(), long_run.impact.round(2).tolist()
    ([[0.45, 0.0], [0.17, 0.64]], [[0.44, -0.11], [0.31, 0.58]])
    >>> covariances = [s.impact @ s.impact.T for s in (recursive, long_run)]
    >>> bool(np.allclose(covariances[0], res.sigma_u)), bool(np.allclose(*covariances))
    (True, True)
    >>> permanent = [bool(abs(s.long_run_impact[0, 1]) < 1e-10) for s in (recursive, long_run)]
    >>> permanent
    [False, True]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from ..._core import (
    _LOG_2PI,
    _PARTIAL_IDENTIFICATION_NOTE,
    _UNIT_SHOCK_NOTE,
    ClosedSystemResult,
    SummaryTable,
    _long_run_matrix,
    _lower_cholesky,
    _validate_impact_pattern,
    _validate_ordering,
)
from ..._internals import (
    _IdentificationModel,
    _maximize_likelihood,
    _MixedHorizonObjective,
    _ShortRunObjective,
    _solve,
    _SummaryMixin,
)
from ...exceptions import NumericalError, SpecificationError

__all__ = ["LongRunSVAR", "MixedSVAR", "RecursiveSVAR", "SVARResult", "ShortRunSVAR"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SVARResult(_SummaryMixin):
    """A point-identified structural view of a fitted reduced-form system.

    Carries exactly as many shock columns as the scheme identified: all ``k``
    under a recursive or long-run ordering, one under a proxy instrument. The
    surface is the same either way -- structural impulse responses, variance
    shares, the recovered shock series, and each shock's historical
    contribution -- and every method is computed per identified column, so a
    partially identified result never reports a number its restrictions do
    not support.

    Attributes:
        source: The reduced-form result the identification was applied to.
        impact: ``(k, s)`` impact columns; entry ``[i, j]`` is variable ``i``'s
            response on impact to one standard deviation of shock ``j``.
        shock_names: One label per identified shock column.
        scheme: Short name of the identification scheme.
        restriction: The identifying restriction, stated as a sentence.
        diagnostics: Scheme-specific metadata pairs shown in the summary --
            a proxy's first-stage strength, for instance.
    """

    source: ClosedSystemResult = field(repr=False)
    impact: npt.NDArray[np.float64] = field(repr=False)
    shock_names: tuple[str, ...]
    scheme: str
    restriction: str
    diagnostics: tuple[tuple[str, str], ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form."""
        return self.source.names

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return len(self.source.names)

    @property
    def k_shocks(self) -> int:
        """Number of identified shocks."""
        return len(self.shock_names)

    @property
    def is_complete(self) -> bool:
        """Whether every structural shock is identified."""
        return self.k_shocks == self.k_endog

    @property
    def long_run_impact(self) -> npt.NDArray[np.float64]:
        """Cumulated response of each variable to each identified shock, ``(k, s)``."""
        ma = getattr(self.source, "ma_coefficients", None)
        return _long_run_matrix(self.source.coefficients, ma) @ self.impact

    def irf(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """Structural impulse responses of every variable to the identified shocks.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums, for differenced data and level
                questions.

        Returns:
            An array of shape ``(horizon + 1, k, s)``; entry ``[h, i, j]`` is
            the response of variable ``i`` at lead ``h`` to one standard
            deviation of identified shock ``j``.
        """
        psi = self.source.ma_representation(horizon)
        theta = np.einsum("hik,ks->his", psi, self.impact)
        return np.cumsum(theta, axis=0) if cumulative else theta

    def fevd(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Share of forecast-error variance carried by each identified shock.

        Returns:
            An array of shape ``(horizon + 1, k, s)`` whose entry ``[h, i, j]``
            is the share of variable ``i``'s ``h + 1``-step forecast-error
            variance attributable to shock ``j``. Rows sum to one across
            shocks only when the identification is complete; under partial
            identification the shortfall is the variance the scheme does not
            speak for.
        """
        psi = self.source.ma_representation(horizon)
        theta = np.einsum("hik,ks->his", psi, self.impact)
        explained = np.cumsum(theta**2, axis=0)
        total = np.cumsum(np.einsum("hik,kl,hil->hi", psi, self.source.sigma_u, psi), axis=0)
        return explained / total[:, :, np.newaxis]

    def structural_shocks(self) -> npt.NDArray[np.float64]:
        """The identified shock series, ``(nobs, s)``, unit variance by construction.

        Recovered as ``b_j' Sigma^{-1} u_t``, which is the projection that
        needs only the identified columns: a partially identified scheme can
        recover its own shocks without ever knowing the rest of the impact
        matrix.
        """
        rotated = np.linalg.solve(self.source.sigma_u, self.source.resid.T).T
        return rotated @ self.impact

    def historical_decomposition(self) -> npt.NDArray[np.float64]:
        """Each identified shock's cumulative contribution to each variable.

        Returns:
            An array of shape ``(nobs, k, s)``; entry ``[t, i, j]`` is shock
            ``j``'s contribution to variable ``i`` at time ``t``. Summing over
            identified shocks recovers the full stochastic path only under
            complete identification; the remainder belongs to the shocks the
            scheme leaves unnamed.

        Note:
            Costs ``O(nobs^2)`` moving-average terms, like its reduced-form
            counterpart.
        """
        shocks = self.structural_shocks()
        nobs = shocks.shape[0]
        theta = self.irf(nobs - 1)
        out = np.zeros((nobs, self.k_endog, self.k_shocks), dtype=np.float64)
        for t in range(nobs):
            out[t] = np.einsum("lis,ls->is", theta[: t + 1], shocks[t::-1])
        return out

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = tuple(
            (name, *(f"{self.impact[i, j]:.4f}" for j in range(self.k_shocks)))
            for i, name in enumerate(self.names)
        )
        notes = [self.restriction, _UNIT_SHOCK_NOTE]
        if not self.is_complete:
            notes.append(_PARTIAL_IDENTIFICATION_NOTE)
        return SummaryTable(
            title=f"Structural VAR ({self.scheme})",
            metadata=(
                ("Scheme", self.scheme),
                ("Identified shocks", f"{self.k_shocks} of {self.k_endog}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.source.nobs}"),
                *self.diagnostics,
            ),
            columns=("impact of", *self.shock_names),
            rows=rows,
            notes=tuple(notes),
        )


class RecursiveSVAR(_IdentificationModel[SVARResult]):
    """Recursive identification: a declared causal ordering, Sims (1980).

    The impact matrix is the Cholesky factor of the innovation covariance in
    the declared ordering: the first variable responds to no shock but its own
    on impact, the second to the first and its own, and so on down the
    triangle. This is the same arithmetic the reduced-form ``irf`` performs
    with ``orthogonalized=True`` -- the difference, and the reason this model
    exists, is that here the ordering is an argument someone chose rather than
    an accident of column order.

    Args:
        result: The fitted closed reduced-form result to identify.
        order: The causal ordering, most exogenous first. ``None`` declares
            the ordering to be the variables as they stand.

    Raises:
        SpecificationError: If the result is not a closed system, or the
            ordering is not a permutation of its variable names.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> y = np.diff(rng.standard_normal((121, 2)).cumsum(axis=0) * 0.1, axis=0)
        >>> svar = RecursiveSVAR(VAR(y, order=1).fit()).identify()
        >>> svar.impact.shape
        (2, 2)
    """

    __slots__ = ("_perm",)

    def __init__(self, result: ClosedSystemResult, *, order: Sequence[str] | None = None) -> None:
        """Validate the source system and the declared ordering."""
        super().__init__(result)
        self._perm = _validate_ordering(self.names, order)

    @property
    def ordering(self) -> tuple[str, ...]:
        """The declared causal ordering."""
        return tuple(self.names[i] for i in self._perm)

    def identify(self) -> SVARResult:
        """Factor the innovation covariance in the declared ordering.

        Returns:
            The complete structural result, shock columns in the declared
            ordering.

        Raises:
            NumericalError: If the innovation covariance is not positive
                definite.
        """
        perm = self._perm
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        factor = _lower_cholesky(sigma[np.ix_(perm, perm)], "sigma_u")
        impact = np.empty_like(factor)
        impact[list(perm), :] = factor
        ordering = self.ordering
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=ordering,
            scheme="recursive",
            restriction=(
                "Recursive ordering "
                + " -> ".join(ordering)
                + ": each variable responds on impact only to shocks at or "
                "before its own position. Permuting the ordering changes the "
                "answer; that is the identifying assumption, not a numerical "
                "artifact."
            ),
        )


class LongRunSVAR(_IdentificationModel[SVARResult]):
    r"""Long-run recursive identification, Blanchard-Quah (1989).

    The restriction lives at the infinite horizon: the *cumulated* response
    matrix is lower triangular in the declared ordering, so the first shock is
    the only one with a permanent effect on the first variable, and so on. In
    the bivariate Blanchard-Quah economy -- output growth first, unemployment
    second -- the first shock is supply, the only one that moves the level of
    output forever, and demand is whatever remains. Computed in closed form:
    with :math:`F = A(1)^{-1} M(1)` the long-run impact of the innovations,

    .. math::

       F \Sigma_u F' = \Theta \Theta', \qquad
       \Theta \text{ lower triangular}, \qquad
       B = F^{-1} \Theta,

    the lower Cholesky factor of the long-run covariance is the long-run
    impact of the shocks, and the impact matrix is :math:`F^{-1}` times it.
    The system must be stationary for :math:`F` to exist, and nearly
    nonstationary systems make it ill-conditioned: the result is checked
    against the innovation covariance and refused when it fails to
    reproduce it.

    Args:
        result: The fitted closed, stationary reduced-form result to identify.
        order: The ordering of permanence, most permanent first. ``None``
            declares the variables as they stand.

    Raises:
        SpecificationError: If the result is not a closed system, is not
            stationary, or the ordering is not a permutation of its names.

    Attributes:
        _source: The closed reduced-form result being identified.
        _perm: The ordering as column indices into the variable names.

    See Also:
        * :class:`SVARResult` -- the complete result returned.
        * :class:`RecursiveSVAR` -- the same triangle on the impact matrix.
        * :class:`MixedSVAR` -- zeros split between the two horizons.

    References:
        Blanchard, O. J., & Quah, D. (1989). The dynamic effects of
        aggregate demand and supply disturbances. *American Economic
        Review*, 79(4), 655-673.

    Example:
        A bivariate system built so that the long-run impact matrix of the
        structural shocks is lower triangular: only the first shock moves
        the first variable's level permanently. The long-run factorization
        recovers the impact matrix -- which is not triangular -- and the
        first shock series:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> F = np.array([[1.0, 0.0], [0.5, 1.0]])
        >>> B = (np.eye(2) - A) @ F
        >>> eps = rng.standard_normal((602, 2))
        >>> y = np.zeros((602, 2))
        >>> for t in range(1, 602):
        ...     y[t] = A @ y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("dy", "u")).fit()
        >>> svar = LongRunSVAR(res).identify()
        >>> svar.scheme, svar.shock_names, svar.is_complete
        ('long-run', ('dy', 'u'), True)
        >>> svar.long_run_impact.round(2).tolist()
        [[1.06, -0.0], [0.67, 0.94]]
        >>> svar.impact.round(2).tolist(), B.round(2).tolist()
        ([[0.44, -0.11], [0.31, 0.58]], [[0.45, -0.1], [0.3, 0.6]])
        >>> bool(np.corrcoef(svar.structural_shocks()[:, 0], eps[1:, 0])[0, 1] > 0.95)
        True
    """

    __slots__ = ("_perm",)

    def __init__(self, result: ClosedSystemResult, *, order: Sequence[str] | None = None) -> None:
        """Validate the source system, its stationarity, and the ordering.

        Args:
            result: The fitted closed reduced-form result to identify.
            order: A permutation of the variable names, or ``None``.

        Raises:
            SpecificationError: If the result is not a closed system, its
                companion is explosive or has a unit root, or ``order`` is
                not a permutation of its variable names.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("dy", "u")).fit()
            >>> LongRunSVAR(res, order=("u", "dy"))._perm
            (1, 0)
            >>> x = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     x[t] = 1.05 * x[t - 1] + rng.standard_normal(2)
            >>> explosive = VAR(x, order=1).fit()
            >>> explosive.is_stable
            False
            >>> LongRunSVAR(explosive)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: long-run identification needs a stationary ...
        """
        super().__init__(result)
        if getattr(result, "is_stable", True) is False:
            raise SpecificationError(
                "long-run identification needs a stationary system: an "
                "explosive or unit-root companion has no finite cumulated "
                "response to restrict."
            )
        self._perm = _validate_ordering(self.names, order)

    @property
    def ordering(self) -> tuple[str, ...]:
        """The declared ordering of permanence, most permanent first.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("dy", "u")).fit()
            >>> LongRunSVAR(res).ordering, LongRunSVAR(res, order=("u", "dy")).ordering
            (('dy', 'u'), ('u', 'dy'))
        """
        return tuple(self.names[i] for i in self._perm)

    def identify(self) -> SVARResult:
        """Factor the long-run covariance and map back to impact.

        Returns:
            The complete structural result, shock columns ordered by
            permanence.

        Raises:
            SpecificationError: If the autoregressive polynomial has a unit
                root, so no long-run impact matrix exists.
            NumericalError: If the recovered impact matrix fails to reproduce
                the innovation covariance, which indicates the long-run matrix
                is too ill-conditioned to identify through.

        Example:
            Either ordering reproduces the innovation covariance; the
            long-run matrix is triangular in the declared one:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("dy", "u")).fit()
            >>> first = LongRunSVAR(res).identify()
            >>> second = LongRunSVAR(res, order=("u", "dy")).identify()
            >>> bool(abs(first.long_run_impact[0, 1]) < 1e-10)
            True
            >>> bool(abs(second.long_run_impact[1, 1]) < 1e-10), second.shock_names
            (True, ('u', 'dy'))
            >>> bool(np.allclose(second.impact @ second.impact.T, res.sigma_u))
            True
        """
        perm = self._perm
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        ma = getattr(self.source, "ma_coefficients", None)
        total = _long_run_matrix(self.source.coefficients, ma)
        long_run_cov = total @ sigma @ total.T
        factor = _lower_cholesky(long_run_cov[np.ix_(perm, perm)], "the long-run covariance")
        theta = np.empty_like(factor)
        theta[list(perm), :] = factor
        impact = np.linalg.solve(total, theta)
        tolerance = 1e-8 * max(1.0, float(np.abs(sigma).max()))
        if not np.allclose(impact @ impact.T, sigma, atol=tolerance):
            raise NumericalError(
                "the long-run factorization does not reproduce the innovation "
                "covariance; the long-run matrix is too ill-conditioned to "
                "identify through, which usually means the system is close to "
                "a unit root."
            )
        ordering = self.ordering
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=ordering,
            scheme="long-run",
            restriction=(
                "Long-run triangularity in the ordering "
                + " -> ".join(ordering)
                + ": each shock has no permanent effect on the variables "
                "before its position. The restriction binds cumulated "
                "responses, so on differenced data it restricts levels."
            ),
        )


class ShortRunSVAR(_IdentificationModel[SVARResult]):
    r"""General short-run zero restrictions: the AB model, Amisano-Giannini (1997).

    .. math::

       A u_t = B \varepsilon_t, \qquad
       \mathbb{E}\varepsilon_t \varepsilon_t' = I, \qquad
       \Sigma_u = A^{-1} B B' A^{-\top},

    with :math:`A` carrying the contemporaneous relations among the
    innovations, :math:`B` the loadings of the shocks, and the impact
    matrix :math:`A^{-1} B`. Restrictions are declared cell by cell -- a
    finite entry fixes a coefficient, ``nan`` frees it -- which is what
    lifts the scheme past :class:`RecursiveSVAR`: zeros can sit anywhere,
    not only above a diagonal, at the price of a likelihood search where
    the triangle had a closed form.

    The order condition is enforced at construction: the covariance supplies
    :math:`k(k+1)/2` equations, so at most that many coefficients can be
    free. Fewer means over-identification, and the likelihood-ratio statistic
    against the unrestricted covariance -- the classical over-identification
    test -- is computed and reported on the result. The rank condition has no
    clean a-priori check for arbitrary patterns; it shows up at ``identify``
    time as a just-identified model that cannot reproduce the covariance, and
    is reported as exactly that.

    Args:
        result: The fitted closed reduced-form result to identify.
        a: ``(k, k)`` pattern for the contemporaneous relations, ``nan`` for a
            free coefficient. ``None`` fixes ``A`` to the identity, the
            B-model of Bernanke (1986). Diagonal entries must be fixed --
            conventionally one -- because a free diagonal trades scale with
            ``B`` and nothing identifies the split.
        b: ``(k, k)`` pattern for the shock loadings, ``nan`` for a free
            coefficient. ``None`` frees the diagonal and fixes the rest to
            zero, the A-model in which each equation has its own shock.
        shock_names: One label per shock column. Defaults to the variable
            names.

    Raises:
        SpecificationError: If the result is not a closed system, both
            patterns are omitted, a pattern is malformed, the diagonal of
            ``a`` is free, no coefficient is free, or the order condition
            fails.

    Attributes:
        _source: The closed reduced-form result being identified.
        _a_base: The fixed values of ``A`` with zeros in the free cells.
        _a_free: Row-major coordinates of the free cells of ``A``.
        _b_base: The fixed values of ``B`` with zeros in the free cells.
        _b_free: Row-major coordinates of the free cells of ``B``.
        _overid_df: Restrictions beyond exact identification.
        _labels: The shock labels.

    See Also:
        * :class:`SVARResult` -- the complete result returned, with the
          over-identification test in its diagnostics.
        * :class:`RecursiveSVAR` -- the triangular special case, in closed
          form.
        * :class:`MixedSVAR` -- when some zeros belong at the long run.

    References:
        Amisano, G., & Giannini, C. (1997). *Topics in Structural VAR
        Econometrics* (2nd ed.). Springer.

        Bernanke, B. S. (1986). Alternative explanations of the
        money-income correlation. *Carnegie-Rochester Conference Series on
        Public Policy*, 25, 49-99.

    Example:
        A three-variable B-model. A lower-triangular pattern is exactly
        identified and reproduces the recursive factorization; fixing one
        more cell over-identifies the structure, and the likelihood-ratio
        test says whether the data object:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[0.45, -0.1, 0.0], [0.27, 0.58, -0.1], [0.11, 0.14, 0.7]])
        >>> eps = rng.standard_normal((802, 3))
        >>> y = np.zeros((802, 3))
        >>> for t in range(1, 802):
        ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("prod", "hours", "rate")).fit()
        >>> nan = np.nan
        >>> exact = ShortRunSVAR(res, b=[[nan, 0, 0], [nan, nan, 0], [nan, nan, nan]])
        >>> exact.n_free, exact.overidentifying_restrictions
        (6, 0)
        >>> svar = exact.identify()
        >>> dict(svar.diagnostics)["Identification"], svar.scheme
        ('exact', 'short-run')
        >>> bool(np.allclose(svar.impact, RecursiveSVAR(res).identify().impact))
        True
        >>> over = ShortRunSVAR(res, b=[[nan, 0, 0], [nan, nan, 0], [nan, 0, nan]]).identify()
        >>> {key: value for key, value in over.diagnostics if key.startswith("Over-ID")}
        {'Over-ID restrictions': '1', 'Over-ID LR': '1.251', 'Over-ID p-value': '0.2634'}
    """

    __slots__ = ("_a_base", "_a_free", "_b_base", "_b_free", "_labels", "_overid_df")

    def __init__(
        self,
        result: ClosedSystemResult,
        *,
        a: npt.ArrayLike | None = None,
        b: npt.ArrayLike | None = None,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system, the patterns, and the order condition.

        Args:
            result: The fitted closed reduced-form result to identify.
            a: The ``(k, k)`` pattern on ``A``, or ``None`` for the identity.
            b: The ``(k, k)`` pattern on ``B``, or ``None`` for a free
                diagonal with zeros elsewhere.
            shock_names: ``k`` unique labels, or ``None`` for the variable
                names.

        Raises:
            SpecificationError: If the result is not a closed system, both
                patterns are ``None``, a pattern has the wrong shape or an
                infinity, a diagonal cell of ``a`` is free, every cell is
                fixed, more cells are free than ``k(k+1)/2``, or the labels
                are not ``k`` unique strings.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("r", "x")).fit()
            >>> nan = np.nan
            >>> model = ShortRunSVAR(res, a=[[1.0, 0.0], [nan, 1.0]])
            >>> model._a_free, model._b_free, model._overid_df, model._labels
            (((1, 0),), ((0, 0), (1, 1)), 0, ('r', 'x'))
            >>> ShortRunSVAR(res)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: declare at least one pattern: with neither ...
            >>> ShortRunSVAR(res, a=[[nan, 0.0], [0.0, 1.0]])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: the diagonal of a must be fixed ...
            >>> ShortRunSVAR(res, b=[[1.0, 0.0], [0.0, 1.0]])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: every coefficient is fixed; there is ...
            >>> ShortRunSVAR(res, a=[[1.0, nan], [nan, 1.0]], b=[[nan, nan], [nan, nan]])
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: the order condition fails: 6 free ...
        """
        super().__init__(result)
        k = self.k_endog
        if a is None and b is None:
            raise SpecificationError(
                "declare at least one pattern: with neither a nor b restricted "
                "there is nothing here that RecursiveSVAR does not already do "
                "in closed form."
            )
        self._a_base, self._a_free = _validate_impact_pattern(
            a, size=k, label="a", default_diagonal=1.0
        )
        if any(i == j for i, j in self._a_free):
            raise SpecificationError(
                "the diagonal of a must be fixed (conventionally 1): a free "
                "diagonal trades scale with b, and nothing identifies the split."
            )
        b_pattern = np.where(np.eye(k) > 0.0, np.nan, 0.0) if b is None else b
        self._b_base, self._b_free = _validate_impact_pattern(
            b_pattern, size=k, label="b", default_diagonal=None
        )
        free = len(self._a_free) + len(self._b_free)
        capacity = k * (k + 1) // 2
        if free < 1:
            raise SpecificationError(
                "every coefficient is fixed; there is nothing to estimate and "
                "the declared structure either reproduces the covariance or "
                "contradicts it."
            )
        if free > capacity:
            raise SpecificationError(
                f"the order condition fails: {free} free coefficients against "
                f"the {capacity} equations the innovation covariance supplies. "
                "Fix at least the difference."
            )
        self._overid_df = capacity - free
        if shock_names is None:
            self._labels = self.names
        else:
            resolved = tuple(str(name) for name in shock_names)
            if len(resolved) != k or len(set(resolved)) != k:
                raise SpecificationError(f"shock_names must be {k} unique labels; got {resolved}.")
            self._labels = resolved

    @property
    def n_free(self) -> int:
        """Free coefficients across both matrices.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 3)), order=1).fit()
            >>> ShortRunSVAR(res, a=[[1, 0, 0], [np.nan, 1, 0], [np.nan, np.nan, 1]]).n_free
            6
        """
        return len(self._a_free) + len(self._b_free)

    @property
    def overidentifying_restrictions(self) -> int:
        """Restrictions beyond the count needed for exact identification.

        ``k(k+1)/2`` minus the free coefficients; zero means exactly
        identified and no test.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 3)), order=1).fit()
            >>> ShortRunSVAR(res, a=np.eye(3)).overidentifying_restrictions
            3
        """
        return self._overid_df

    def identify(self) -> SVARResult:
        """Estimate the free coefficients by maximum likelihood.

        The likelihood is evaluated at the result's reported innovation
        covariance, so a just-identified structure reproduces exactly the
        matrix every other scheme factors and the whole result surface stays
        internally consistent. Each impact column is signed so its largest
        entry is positive.

        Returns:
            The complete structural result, with the over-identification
            likelihood-ratio test in its diagnostics when restrictions exceed
            the exactly identifying count.

        Raises:
            NumericalError: If a just-identified structure cannot reproduce
                the innovation covariance, which is the rank condition failing
                at this pattern -- the restrictions are arranged so that some
                free coefficient is not pinned down.

        Example:
            The default B-model -- uncorrelated structural shocks loading
            one per equation -- is over-identified by ``k(k-1)/2`` zeros,
            and on correlated innovations the test rejects it:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("r", "x")).fit()
            >>> diagonal = ShortRunSVAR(res, b=[[np.nan, 0.0], [0.0, np.nan]]).identify()
            >>> report = dict(diagonal.diagnostics)
            >>> report["Over-ID restrictions"], float(report["Over-ID p-value"]) < 0.01
            ('1', True)
            >>> bool(np.allclose(diagonal.impact, np.diag(np.sqrt(np.diag(res.sigma_u)))))
            True
            >>> diagonal.restriction[:41]
            'AB model with 4 fixed cells in the contem'
        """
        k = self.k_endog
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        objective = _ShortRunObjective(
            sigma=sigma,
            nobs=int(self.source.nobs),
            a_base=self._a_base,
            a_free=self._a_free,
            b_base=self._b_base,
            b_free=self._b_free,
        )
        (a_hat, b_hat), llf = _maximize_likelihood(objective)
        impact = np.linalg.solve(a_hat, b_hat)
        for j in range(k):
            column = impact[:, j]
            if column[int(np.argmax(np.abs(column)))] < 0.0:
                impact[:, j] = -column

        saturated = float(self.source.nobs) * (
            -0.5 * k * _LOG_2PI - 0.5 * float(np.linalg.slogdet(sigma)[1]) - 0.5 * k
        )
        ratio = max(2.0 * (saturated - llf), 0.0)
        if self._overid_df == 0:
            tolerance = 1e-6 * max(1.0, float(np.abs(sigma).max()))
            if not np.allclose(impact @ impact.T, sigma, atol=tolerance):
                raise NumericalError(
                    "a just-identified pattern failed to reproduce the "
                    "innovation covariance: the rank condition fails at this "
                    "arrangement of zeros, so some free coefficient is not "
                    "pinned down. Rearrange the restrictions."
                )
            diagnostics: tuple[tuple[str, str], ...] = (
                ("Free parameters", f"{self.n_free}"),
                ("Identification", "exact"),
            )
        else:
            pvalue = float(chi2.sf(ratio, self._overid_df))
            diagnostics = (
                ("Free parameters", f"{self.n_free}"),
                ("Over-ID restrictions", f"{self._overid_df}"),
                ("Over-ID LR", f"{ratio:.3f}"),
                ("Over-ID p-value", f"{pvalue:.4f}"),
            )
        fixed_a = k * k - len(self._a_free)
        fixed_b = k * k - len(self._b_free)
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=self._labels,
            scheme="short-run",
            restriction=(
                f"AB model with {fixed_a} fixed cells in the contemporaneous "
                f"relations and {fixed_b} in the shock loadings, "
                f"{self.n_free} coefficients estimated by maximum likelihood. "
                "Where the restrictions exceed the exactly identifying count, "
                "the over-identification test below is the data's verdict on "
                "them; where they do not, the structure is an assumption the "
                "data cannot contradict."
            ),
            diagnostics=diagnostics,
        )


class MixedSVAR(_IdentificationModel[SVARResult]):
    r"""Zero restrictions split across impact and the long run, Galí (1999).

    The scheme that needs both horizons at once: a technology shock is the
    only one moving productivity forever -- a long-run zero -- while a policy
    shock is barred from moving output on impact -- a short-run zero. Neither
    :class:`ShortRunSVAR` nor :class:`LongRunSVAR` can say both; this model
    takes a pattern for each matrix and finds the one factorization
    satisfying every declared cell. The search runs where the problem
    lives, over rotations of the Cholesky factor,

    .. math::

       B = L Q, \qquad Q' Q = I, \qquad
       B_{ij} = c_{ij} \;\text{(impact cells)}, \qquad
       (F B)_{ij} = d_{ij} \;\text{(long-run cells)},

    so the innovation covariance is reproduced identically at every
    candidate and the restrictions are the only thing being solved for. A
    rotation has exactly :math:`k(k-1)/2` degrees of freedom, which is why
    exactly that many restrictions are required -- fewer is an
    under-identified pattern, more is an over-identified one whose
    restricted covariance estimation this model deliberately does not
    attempt.

    Args:
        result: The fitted closed, stationary reduced-form result to identify.
        impact: ``(k, k)`` pattern on the impact matrix, ``nan`` for a free
            cell and a finite value -- almost always zero -- for a fixed one.
            ``None`` fixes nothing on impact.
        long_run: ``(k, k)`` pattern on the cumulated long-run matrix, same
            convention. Required: with nothing fixed at the long run,
            :class:`ShortRunSVAR` is that model.
        shock_names: One label per shock column. Defaults to the variable
            names.

    Raises:
        SpecificationError: If the result is not a closed system or not
            stationary, the long-run pattern is omitted, a pattern is
            malformed, or the restriction count differs from
            ``k (k - 1) / 2``.

    Attributes:
        _source: The closed reduced-form result being identified.
        _impact_cells: The fixed impact cells as ``(row, column, value)``.
        _long_cells: The fixed long-run cells as ``(row, column, value)``.
        _labels: The shock labels.

    See Also:
        * :class:`SVARResult` -- the complete result returned.
        * :class:`LongRunSVAR` -- all zeros at the long run, in closed form.
        * :class:`ShortRunSVAR` -- all zeros on impact, with
          over-identification allowed.

    References:
        Galí, J. (1999). Technology, employment, and the business cycle: Do
        technology shocks explain aggregate fluctuations? *American
        Economic Review*, 89(1), 249-271.

    Example:
        Three variables, three restrictions: the technology shock alone
        moves productivity in the long run (two long-run zeros) and the
        policy shock cannot move productivity on impact (one short-run
        zero). The system is built with exactly those properties, and the
        rotation recovers the impact matrix and the shocks:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
        >>> F = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.3, 0.2, 1.0]])
        >>> B = (np.eye(3) - A) @ F
        >>> eps = rng.standard_normal((802, 3))
        >>> y = np.zeros((802, 3))
        >>> for t in range(1, 802):
        ...     y[t] = A @ y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("prod", "hours", "rate")).fit()
        >>> nan = np.nan
        >>> model = MixedSVAR(
        ...     res,
        ...     impact=[[nan, nan, 0.0], [nan, nan, nan], [nan, nan, nan]],
        ...     long_run=[[nan, 0.0, 0.0], [nan, nan, nan], [nan, nan, nan]],
        ...     shock_names=("technology", "demand", "policy"),
        ... )
        >>> model.n_restrictions
        3
        >>> svar = model.identify()
        >>> svar.scheme, svar.shock_names, dict(svar.diagnostics)["Identification"]
        ('mixed', ('technology', 'demand', 'policy'), 'exact')
        >>> svar.impact.round(2).tolist()
        [[0.45, -0.09, 0.0], [0.26, 0.57, -0.05], [0.08, 0.07, 0.73]]
        >>> svar.long_run_impact[0].round(2).tolist()
        [1.02, 0.0, -0.0]
        >>> bool(np.corrcoef(svar.structural_shocks()[:, 0], eps[1:, 0])[0, 1] > 0.95)
        True
    """

    __slots__ = ("_impact_cells", "_labels", "_long_cells")

    def __init__(
        self,
        result: ClosedSystemResult,
        *,
        impact: npt.ArrayLike | None = None,
        long_run: npt.ArrayLike | None = None,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system, both patterns, and the order condition.

        Args:
            result: The fitted closed reduced-form result to identify.
            impact: The ``(k, k)`` impact pattern, or ``None``.
            long_run: The ``(k, k)`` long-run pattern; required, with at
                least one fixed cell.
            shock_names: ``k`` unique labels, or ``None`` for the variable
                names.

        Raises:
            SpecificationError: If the result is not a closed system, its
                companion is explosive or has a unit root, ``long_run`` is
                omitted or fixes nothing, a pattern has the wrong shape or
                an infinity, the fixed cells do not number exactly
                ``k(k-1)/2``, or the labels are not ``k`` unique strings.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("dy", "u")).fit()
            >>> nan = np.nan
            >>> model = MixedSVAR(res, long_run=[[nan, 0.0], [nan, nan]])
            >>> model._impact_cells, model._long_cells, model._labels
            ((), ((0, 1, 0.0),), ('dy', 'u'))
            >>> MixedSVAR(res, impact=[[nan, 0.0], [nan, nan]])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: long_run must fix at least one cell; ...
            >>> MixedSVAR(res, impact=[[nan, 0.0], [nan, nan]], long_run=[[nan, 0.0], [nan, nan]])
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a rotation-solved mixed pattern needs ...
        """
        super().__init__(result)
        if getattr(result, "is_stable", True) is False:
            raise SpecificationError(
                "mixed-horizon identification needs a stationary system: an "
                "explosive or unit-root companion has no finite cumulated "
                "response to restrict."
            )
        k = self.k_endog
        if long_run is None:
            raise SpecificationError(
                "long_run must fix at least one cell; with restrictions on "
                "impact alone, ShortRunSVAR is that model."
            )
        self._impact_cells = self._fixed_cells(impact, k, "impact")
        self._long_cells = self._fixed_cells(long_run, k, "long_run")
        if not self._long_cells:
            raise SpecificationError(
                "long_run must fix at least one cell; with restrictions on "
                "impact alone, ShortRunSVAR is that model."
            )
        declared = len(self._impact_cells) + len(self._long_cells)
        needed = k * (k - 1) // 2
        if declared != needed:
            raise SpecificationError(
                f"a rotation-solved mixed pattern needs exactly {needed} "
                f"restrictions for {k} variables; got {declared}. Fewer leaves "
                "the rotation under-determined; more is an over-identified "
                "pattern, whose restricted covariance estimation this model "
                "does not attempt."
            )
        if shock_names is None:
            self._labels = self.names
        else:
            resolved = tuple(str(name) for name in shock_names)
            if len(resolved) != k or len(set(resolved)) != k:
                raise SpecificationError(f"shock_names must be {k} unique labels; got {resolved}.")
            self._labels = resolved

    @staticmethod
    def _fixed_cells(
        pattern: npt.ArrayLike | None, size: int, label: str
    ) -> tuple[tuple[int, int, float], ...]:
        """The declared restrictions of one pattern, as (row, column, value).

        Args:
            pattern: The ``(size, size)`` pattern, or ``None`` for no
                restrictions.
            size: System dimension.
            label: Which pattern this is, for error messages.

        Returns:
            The fixed cells in row-major order.

        Raises:
            SpecificationError: If the pattern has the wrong shape or
                contains an infinity.

        Example:
            >>> import numpy as np
            >>> MixedSVAR._fixed_cells(None, 2, "impact")
            ()
            >>> MixedSVAR._fixed_cells([[np.nan, 0.0], [1.5, np.nan]], 2, "impact")
            ((0, 1, 0.0), (1, 0, 1.5))
        """
        if pattern is None:
            return ()
        base, free = _validate_impact_pattern(pattern, size=size, label=label)
        free_set = set(free)
        return tuple(
            (i, j, float(base[i, j]))
            for i in range(size)
            for j in range(size)
            if (i, j) not in free_set
        )

    @property
    def n_restrictions(self) -> int:
        """Declared restrictions across both horizons, always ``k(k-1)/2``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 3)), order=1).fit()
            >>> nan = np.nan
            >>> pattern = [[nan, 0.0, 0.0], [nan, nan, 0.0], [nan, nan, nan]]
            >>> MixedSVAR(res, long_run=pattern).n_restrictions
            3
        """
        return len(self._impact_cells) + len(self._long_cells)

    def identify(self) -> SVARResult:
        """Solve for the rotation satisfying every declared cell.

        Both components of the orthogonal group are searched -- rotations
        directly, reflections through a fixed sign flip of the factor -- so a
        pattern with nonzero fixed values is reachable wherever it lives.
        Columns whose fixed cells are all zero are signed so their largest
        entry is positive; a column with a nonzero fixed value keeps the
        sign that value implies.

        Returns:
            The complete structural result.

        Raises:
            SpecificationError: If the autoregressive polynomial has a unit
                root, so no long-run matrix exists.
            NumericalError: If no rotation satisfies the restrictions, which
                is the rank condition failing at this pattern -- the declared
                cells are arranged so that no factorization of this covariance
                can honor all of them at once.

        Example:
            A single long-run zero in two variables is the Blanchard-Quah
            triangle, and the rotation lands on the closed form; a fixed
            impact value no factorization can reach is refused:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("dy", "u")).fit()
            >>> nan = np.nan
            >>> mixed = MixedSVAR(res, long_run=[[nan, 0.0], [nan, nan]]).identify()
            >>> bool(np.allclose(mixed.impact, LongRunSVAR(res).identify().impact))
            True
            >>> dict(mixed.diagnostics)["At the long run"], mixed.restriction[:27]
            ('1', '0 cells fixed on impact and')
            >>> MixedSVAR(res, impact=[[nan, 5.0], [nan, nan]], long_run=[[nan, nan], [nan, 0.0]])
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a rotation-solved mixed pattern needs ...
            >>> MixedSVAR(res, long_run=[[nan, 5.0], [nan, nan]]).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: no factorization of the innovation covariance ...
        """
        k = self.k_endog
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        ma = getattr(self.source, "ma_coefficients", None)
        total = _long_run_matrix(self.source.coefficients, ma)
        factor = _lower_cholesky(sigma, "sigma_u")
        reflect = np.eye(k, dtype=np.float64)
        reflect[-1, -1] = -1.0

        best_impact: npt.NDArray[np.float64] | None = None
        best_violation = np.inf
        for base in (factor, factor @ reflect):
            objective = _MixedHorizonObjective(
                impact_factor=base,
                long_factor=total @ base,
                impact_cells=self._impact_cells,
                long_cells=self._long_cells,
            )
            rotation, violation = _solve(objective)
            if violation < best_violation:
                best_violation = violation
                best_impact = base @ rotation
        assert best_impact is not None
        tolerance = 1e-8 * max(1.0, float(np.abs(sigma).max()))
        if best_violation > tolerance:
            raise NumericalError(
                "no factorization of the innovation covariance honors every "
                f"declared cell (best violation {best_violation:.3e}): the "
                "rank condition fails at this pattern. Rearrange the "
                "restrictions across the two horizons."
            )
        impact = best_impact
        for j in range(k):
            fixed_values = [
                value for i, c, value in (*self._impact_cells, *self._long_cells) if c == j
            ]
            if all(value == 0.0 for value in fixed_values):
                column = impact[:, j]
                if column[int(np.argmax(np.abs(column)))] < 0.0:
                    impact[:, j] = -column
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=self._labels,
            scheme="mixed",
            restriction=(
                f"{len(self._impact_cells)} cells fixed on impact and "
                f"{len(self._long_cells)} on the cumulated long-run matrix, "
                "solved jointly over rotations of the Cholesky factor. The "
                "covariance is reproduced identically; the restrictions alone "
                "chose the rotation, and they are assumptions the data cannot "
                "contradict at exact identification."
            ),
            diagnostics=(
                ("Restrictions", f"{self.n_restrictions}"),
                ("On impact", f"{len(self._impact_cells)}"),
                ("At the long run", f"{len(self._long_cells)}"),
                ("Max violation", f"{best_violation:.2e}"),
                ("Identification", "exact"),
            ),
        )
