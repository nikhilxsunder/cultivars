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

"""Zero-restriction identification: triangularity on impact or in the long run.

A reduced-form innovation covariance admits infinitely many factorizations,
and every one of them tells a different economic story. The two models here
choose by the oldest device in the literature: zeros arranged as a triangle,
imposed at one of two horizons. :class:`RecursiveSVAR` puts the triangle on
*impact* -- a declared causal ordering in which each variable responds
contemporaneously only to shocks at or before its own position, Sims (1980).
:class:`LongRunSVAR` puts it at the *infinite horizon* -- an ordering of
permanence in which each shock has no cumulated effect on the variables before
its position, Blanchard-Quah (1989).

Both are complete identifications computed in closed form, and both construct
from a fitted closed reduced-form result rather than from data: the reduced
form supplies every estimable quantity, and what these models add is exactly
the restriction, declared as an argument someone chose.

References:
    Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1).
    Blanchard, O. J., & Quah, D. (1989). The dynamic effects of aggregate
        demand and supply disturbances. *American Economic Review*, 79(4).
    Kilian, L., & Lutkepohl, H. (2017). *Structural Vector Autoregressive
        Analysis*. Cambridge University Press.
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
    _ShortRunObjective,
    _SummaryMixin,
)
from ...exceptions import NumericalError, SpecificationError



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
        total = np.cumsum(
            np.einsum("hik,kl,hil->hi", psi, self.source.sigma_u, psi), axis=0
        )
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

    def __init__(
        self, result: ClosedSystemResult, *, order: Sequence[str] | None = None
    ) -> None:
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
    """Long-run recursive identification, Blanchard-Quah (1989).

    The restriction lives at the infinite horizon: the *cumulated* response
    matrix is lower triangular in the declared ordering, so the first shock is
    the only one with a permanent effect on the first variable, and so on. In
    the bivariate Blanchard-Quah economy -- output growth first, unemployment
    second -- the first shock is supply, the only one that moves the level of
    output forever, and demand is whatever remains.

    Computed in closed form: with ``F`` the long-run impact of the
    innovations, the lower Cholesky factor of ``F Sigma F'`` is the long-run
    impact of the shocks, and the impact matrix is ``F^{-1}`` times it.

    Args:
        result: The fitted closed, stationary reduced-form result to identify.
        order: The ordering of permanence, most permanent first. ``None``
            declares the variables as they stand.

    Raises:
        SpecificationError: If the result is not a closed system, is not
            stationary, or the ordering is not a permutation of its names.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> y = np.diff(rng.standard_normal((121, 2)).cumsum(axis=0) * 0.1, axis=0)
        >>> svar = LongRunSVAR(VAR(y, order=1).fit()).identify()
        >>> bool(abs(svar.long_run_impact[0, 1]) < 1e-10)
        True
    """

    __slots__ = ("_perm",)

    def __init__(
        self, result: ClosedSystemResult, *, order: Sequence[str] | None = None
    ) -> None:
        """Validate the source system, its stationarity, and the ordering."""
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
        """The declared ordering of permanence."""
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
        """
        perm = self._perm
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        ma = getattr(self.source, "ma_coefficients", None)
        total = _long_run_matrix(self.source.coefficients, ma)
        long_run_cov = total @ sigma @ total.T
        factor = _lower_cholesky(
            long_run_cov[np.ix_(perm, perm)], "the long-run covariance"
        )
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
    """General short-run zero restrictions: the AB model, Amisano-Giannini (1997).

    ``A u_t = B e_t`` with unit-variance shocks: ``A`` carries the
    contemporaneous relations among the innovations, ``B`` the loadings of the
    shocks, and the impact matrix is ``A^{-1} B``. Restrictions are declared
    cell by cell -- a finite entry fixes a coefficient, ``nan`` frees it --
    which is what lifts the scheme past :class:`RecursiveSVAR`: zeros can sit
    anywhere, not only above a diagonal, at the price of a likelihood search
    where the triangle had a closed form.

    The order condition is enforced at construction: the covariance supplies
    ``k (k + 1) / 2`` equations, so at most that many coefficients can be
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

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> y = np.diff(rng.standard_normal((201, 2)).cumsum(axis=0) * 0.1, axis=0)
        >>> res = VAR(y, order=1).fit()
        >>> pattern = np.array([[np.nan, 0.0], [np.nan, np.nan]])
        >>> svar = ShortRunSVAR(res, b=pattern).identify()
        >>> svar.impact.shape
        (2, 2)
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
        """Validate the source system, the patterns, and the order condition."""
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
        b_pattern = (
            np.where(np.eye(k) > 0.0, np.nan, 0.0) if b is None else b
        )
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
                raise SpecificationError(
                    f"shock_names must be {k} unique labels; got {resolved}."
                )
            self._labels = resolved

    @property
    def n_free(self) -> int:
        """Free coefficients across both matrices."""
        return len(self._a_free) + len(self._b_free)

    @property
    def overidentifying_restrictions(self) -> int:
        """Restrictions beyond the count needed for exact identification."""
        return self._overid_df

    def identify(self) -> SVARResult:
        """Estimate the free coefficients by maximum likelihood.

        The likelihood is evaluated at the result's reported innovation
        covariance, so a just-identified structure reproduces exactly the
        matrix every other scheme factors and the whole result surface stays
        internally consistent.

        Returns:
            The complete structural result, with the over-identification
            likelihood-ratio test in its diagnostics when restrictions exceed
            the exactly identifying count.

        Raises:
            NumericalError: If a just-identified structure cannot reproduce
                the innovation covariance, which is the rank condition failing
                at this pattern -- the restrictions are arranged so that some
                free coefficient is not pinned down.
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
            -0.5 * k * _LOG_2PI
            - 0.5 * float(np.linalg.slogdet(sigma)[1])
            - 0.5 * k
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
