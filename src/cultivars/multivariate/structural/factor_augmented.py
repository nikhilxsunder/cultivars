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
r"""Structural identification on a factor block: one shock, hundreds of answers.

A factor-augmented VAR is two equations,

.. math::

   \begin{pmatrix} F_t \\ Y_t \end{pmatrix}
   = \sum_{i=1}^{p} A_i \begin{pmatrix} F_{t-i} \\ Y_{t-i} \end{pmatrix} + u_t,
   \qquad
   x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t,

and identifying it is identifying the first: the factor VAR is a closed
reduced form like any other, and any scheme in this package factorizes its
:math:`u_t = B \varepsilon_t`. What earns this row its own model is what
happens next. The observation equation maps the identified factor-space
responses :math:`\Psi_h B` onto every series in the informational panel as
:math:`\Lambda \Psi_h B`, in each series' own units, which is the entire
reason the FAVAR exists -- one identification, answered by hundreds of
responses -- and the map is rotation-invariant, because principal components
recover the factor space only up to rotation while loadings and factors
rotate together (Bernanke, Boivin and Eliasz 2005).

Two commitments shape the surface. First, the declaration lives in one
place. By default the model runs the paper's own scheme -- recursive, with
the observed block ordered last, so the policy shock moves no *factor*
within the period (the panel still responds on impact through
:math:`\Lambda^Y`, and the result says so). Any other scheme rides in
through the same door: identify the factor VAR with whichever model states
the restriction you believe, and hand the result here to be mapped; the
model checks by identity that it was built from this FAVAR's own factor VAR
and then only maps it, so the restriction note on the summary is the
scheme's own. Second, the condition the default scheme depends on is
disclosed rather than assumed. The recursive ordering is clean only if the
factors were purged of the observed block's contemporaneous influence --
the slow-variable cleaning of the reduced form -- and a result built on an
uncleaned fit carries a contamination note on every summary.

Layout. :class:`FactorAugmentedSVAR` is an ``_IdentificationModel`` from
``_internals`` whose source is ``favar.factors``, the
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
on the augmented block; it delegates the default scheme to
:class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
and packages the result with the reduced form into
:class:`FactorAugmentedSVARResult`, which owns the observation-equation
map through the FAVAR's ``loadings`` and standardization scales. The
reduced form is :mod:`~cultivars.multivariate.large_dim.factor_augmented`;
the dependency runs one way, structural on large-dimensional, never back.
The same packaging pattern over regimes is
:class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingSVAR`.

References:
    Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects
    of monetary policy: A factor-augmented vector autoregressive (FAVAR)
    approach. *Quarterly Journal of Economics*, 120(1), 387-422.

    Stock, J. H., & Watson, M. W. (2016). Dynamic factor models,
    factor-augmented vector autoregressions, and structural vector
    autoregressions in macroeconomics. In *Handbook of Macroeconomics*
    (Vol. 2A, pp. 415-525). Elsevier.

Example:
    Thirty indicators on one factor and a policy rate that reacts to it,
    with half the panel declared slow so the recursive default is clean.
    The policy shock leaves the factor untouched on impact and still moves
    the panel through the observed block's own loading; the factor shock
    reaches every series with the sign and size of its loading:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
    >>> rng = np.random.default_rng(0)
    >>> factor = np.zeros(300)
    >>> for t in range(1, 300):
    ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
    >>> weights = rng.uniform(0.5, 1.5, 30)
    >>> panel = np.outer(factor, weights) + 0.3 * rng.standard_normal((300, 30))
    >>> policy = 0.4 * factor + rng.standard_normal(300)
    >>> names = [f"x{i + 1}" for i in range(30)]
    >>> res = FAVAR(
    ...     panel, policy[:, None], n_factors=1, order=1,
    ...     slow=names[:15], panel_names=names, observed_names=("ffr",),
    ... ).fit()
    >>> svar = FactorAugmentedSVAR(res).identify()
    >>> res.cleaned, svar.shock_names, svar.irf(12).shape
    (True, ('f1', 'ffr'), (13, 30, 2))
    >>> float(svar.factor_irf(0)[0][0, 1]), bool(np.abs(svar.impact[:, 1]).max() > 0)
    (0.0, True)
    >>> bool(abs(np.corrcoef(svar.impact[:, 0], weights)[0, 1]) > 0.99)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import StructuralResult, SummaryTable
from ..._internals import _IdentificationModel, _SummaryMixin
from ...exceptions import SpecificationError
from ..large_dim.factor_augmented import FAVARResult
from .zero_restrictions import RecursiveSVAR

__all__ = ["FactorAugmentedSVAR", "FactorAugmentedSVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FactorAugmentedSVARResult(_SummaryMixin):
    r"""An identified factor system, answered at the scale of the panel.

    The factor VAR on :math:`(F_t, Y_t)` is identified like any closed
    system, :math:`u_t = B \varepsilon_t`; the observation equation
    :math:`x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t` then carries every
    identified factor-space response onto the panel,

    .. math::

       \frac{\partial x_{t+h}}{\partial \varepsilon_t}
       = \operatorname{diag}(s)\, \Lambda\, \Psi_h B,

    with :math:`\Lambda = (\Lambda^F, \Lambda^Y)` the loadings on the
    standardized panel and :math:`s` the standardization scales, so the
    responses come back in each series' own units.

    Composition on both sides: :attr:`structural` is the factor-space
    identification with its whole surface -- impact matrix, factor impulse
    responses, variance shares, recovered shocks -- and :attr:`favar` is the
    reduced form with its loadings and fit. This object owns the join: the
    observation-equation map from identified factor shocks to responses of
    every panel series.

    Note:
        The map is rotation-invariant. Principal components recover the
        factor space only up to rotation, and the loadings rotate with the
        factors, so :math:`\Lambda \Psi_h B` does not depend on the factor
        basis even though a factor-by-factor reading of :meth:`factor_irf`
        does. The panel response to the observed block's own shock is not
        zero on impact: the observed variables enter the observation
        equation directly through :math:`\Lambda^Y`, and the recursive
        default only keeps that shock out of the *factors* within the
        period.

    Attributes:
        favar: The fitted factor-augmented reduced form.
        structural: The identification of its factor VAR.

    See Also:
        * :class:`FactorAugmentedSVAR` -- the model whose ``identify()``
          returns this record.
        * :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVARResult`
          -- the reduced form, with the loadings and the standardization.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
          -- the factor-space result under the default scheme.

    References:
        Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
        effects of monetary policy: A factor-augmented vector autoregressive
        (FAVAR) approach. *Quarterly Journal of Economics*, 120(1), 387-422.

    Example:
        Thirty series loading on one AR(1) factor and a policy rate that
        reacts to it. The panel impact of the factor shock is the loadings
        pattern in each series' units, and the panel responses are the
        factor-space responses pushed through the observation equation:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
        >>> rng = np.random.default_rng(0)
        >>> factor = np.zeros(300)
        >>> for t in range(1, 300):
        ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
        >>> weights = rng.uniform(0.5, 1.5, 30)
        >>> panel = np.outer(factor, weights) + 0.3 * rng.standard_normal((300, 30))
        >>> policy = 0.4 * factor + rng.standard_normal(300)
        >>> res = FAVAR(panel, policy[:, None], n_factors=1, order=1, observed_names=("ffr",)).fit()
        >>> svar = FactorAugmentedSVAR(res).identify()
        >>> svar.n_series, svar.shock_names, svar.impact.shape, svar.irf(8).shape
        (30, ('f1', 'ffr'), (30, 2), (9, 30, 2))
        >>> bool(abs(np.corrcoef(svar.impact[:, 0], weights)[0, 1]) > 0.99)
        True
        >>> through = (res.loadings @ svar.factor_irf(0)[0]) * res._scales[:, None]
        >>> bool(np.allclose(svar.impact, through))
        True
    """

    favar: FAVARResult = field(repr=False)
    """The fitted factor-augmented reduced form, with loadings and scales. Kept out of the repr."""

    structural: StructuralResult = field(repr=False)
    """The factor-space identification of ``favar.factors``. Kept out of the repr."""

    @property
    def panel_names(self) -> tuple[str, ...]:
        """One label per panel series.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1, panel_names=list("abcde")).fit()
            >>> FactorAugmentedSVAR(res).identify().panel_names
            ('a', 'b', 'c', 'd', 'e')
        """
        return self.favar.panel_names

    @property
    def n_series(self) -> int:
        """Number of panel series.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1).fit()
            >>> FactorAugmentedSVAR(res).identify().n_series
            5
        """
        return self.favar.n_series

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Labels of the identified shocks, from the factor-space result.

        ``shock1 ... shocks`` when the supplied identification does not
        name its shocks.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=2, order=1, observed_names=("r",)).fit()
            >>> FactorAugmentedSVAR(res).identify().shock_names
            ('f1', 'f2', 'r')
        """
        names = getattr(self.structural, "shock_names", None)
        if names is None:
            return tuple(f"shock{j + 1}" for j in range(self.impact.shape[1]))
        return tuple(names)

    @property
    def impact(self) -> npt.NDArray[np.float64]:
        """Impact responses of every panel series, ``(n_series, s)``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1).fit()
            >>> svar = FactorAugmentedSVAR(res).identify()
            >>> svar.impact.shape, bool(np.allclose(svar.impact, svar.irf(0)[0]))
            ((5, 2), True)
        """
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

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1).fit()
            >>> svar = FactorAugmentedSVAR(res).identify()
            >>> plain, running = svar.irf(4), svar.irf(4, cumulative=True)
            >>> plain.shape, bool(np.allclose(running, np.cumsum(plain, axis=0)))
            ((5, 5, 2), True)
            >>> through = np.einsum("nk,hks->hns", res.loadings, svar.factor_irf(4))
            >>> bool(np.allclose(plain, through * res._scales[None, :, None]))
            True
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

        Example:
            Under the default recursive scheme with the observed block last,
            the observed block's shock moves no factor on impact:

            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=2, order=1).fit()
            >>> svar = FactorAugmentedSVAR(res).identify()
            >>> on_impact = svar.factor_irf(3)[0]
            >>> on_impact.shape, on_impact[:2, 2].tolist()
            ((3, 3), [0.0, 0.0])
        """
        return self.structural.irf(horizon, cumulative=cumulative)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The factor-space impact matrix as the table, the scheme and the
        panel size in the metadata, the scheme's own restriction note first,
        then the panel-scale and rotation-invariance notes, and a
        contamination warning when the reduced form ran without
        slow-variable cleaning.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> names = list("abcde")
            >>> raw = FAVAR(panel, policy, n_factors=1, order=1, panel_names=names).fit()
            >>> table = FactorAugmentedSVAR(raw).identify()._summary_table()
            >>> table.title, table.columns, dict(table.metadata)["Slow-variable cleaning"]
            ('Structural FAVAR (recursive)', ('factor impact of', 'f1', 'y1'), 'no')
            >>> len(table.notes), table.notes[-1][:50]
            (4, 'The reduced form ran without the slow-variable cle')
            >>> cleaned = FAVAR(
            ...     panel, policy, n_factors=1, order=1, panel_names=names, slow=names[:3]
            ... ).fit()
            >>> len(FactorAugmentedSVAR(cleaned).identify()._summary_table().notes)
            3
        """
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

    Constructs with a fitted
    :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR`
    result; the closed system being identified is its factor VAR, and the
    model adds what no factor-space scheme can: the observation-equation map
    onto the panel. Two ways in, one result out. By default ``identify`` runs
    the paper's own scheme -- recursive, observed block last, so the policy
    shock moves no factor within the period. Alternatively, identify the
    factor VAR yourself with any point-identified model in this package and
    pass what it returns as ``structural``; the declaration then lives where
    you stated it, and this model only maps it.

    Attributes:
        _source: The factor VAR being identified, ``favar.factors``.
        _favar: The fitted factor-augmented reduced form.
        _structural: The supplied factor-space identification, or ``None``
            for the recursive route.
        _order: The recursive ordering, or ``None`` for factors first and
            the observed block last.

    See Also:
        * :class:`FactorAugmentedSVARResult` -- the record ``identify()``
          returns.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
          -- the default factor-space scheme.
        * :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingSVAR`
          -- the same packaging pattern over regimes instead of a panel.

    References:
        Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
        effects of monetary policy: A factor-augmented vector autoregressive
        (FAVAR) approach. *Quarterly Journal of Economics*, 120(1), 387-422.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
        >>> rng = np.random.default_rng(0)
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

        The mapping route takes an identification of this fit's own factor
        VAR and refuses one built from another:

        >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
        >>> mine = RecursiveSVAR(res.factors).identify()
        >>> bool(np.allclose(FactorAugmentedSVAR(res, mine).identify().impact, svar.impact))
        True
        >>> other = FAVAR(panel, policy[:, None], n_factors=1, order=2).fit()
        >>> FactorAugmentedSVAR(res, RecursiveSVAR(other.factors).identify())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: the supplied identification was not built from ...
    """

    __slots__ = ("_favar", "_order", "_structural")

    def __init__(
        self,
        favar: FAVARResult,
        structural: StructuralResult | None = None,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the reduced form and the identification route.

        Args:
            favar: The fitted factor-augmented reduced form.
            structural: Optional factor-space identification to map, produced
                by any point-identified model applied to ``favar.factors``.
                When given, ``order`` must be omitted.
            order: The recursive ordering for the default scheme. ``None``
                orders the system as it stands -- factors first, observed
                block last, the Bernanke-Boivin-Eliasz convention.

        Raises:
            SpecificationError: If ``favar`` is not a fitted FAVAR result, both
                ``structural`` and ``order`` are given, or ``structural`` was
                not identified from this FAVAR's own factor VAR.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1).fit()
            >>> FactorAugmentedSVAR(res.factors)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: FactorAugmentedSVAR constructs with a ...
            >>> mine = RecursiveSVAR(res.factors).identify()
            >>> FactorAugmentedSVAR(res, mine, order=("y1", "f1"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: pass either a factor-space identification ...
        """
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

        Example:
            Ordering the observed block first lets its shock move the factor
            on impact; a name outside the factor system is refused:

            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.factor_augmented import FAVAR
            >>> rng = np.random.default_rng(0)
            >>> panel, policy = rng.standard_normal((120, 5)), rng.standard_normal((120, 1))
            >>> res = FAVAR(panel, policy, n_factors=1, order=1, observed_names=("r",)).fit()
            >>> default = FactorAugmentedSVAR(res).identify()
            >>> first = FactorAugmentedSVAR(res, order=("r", "f1")).identify()
            >>> default.shock_names, first.shock_names
            (('f1', 'r'), ('r', 'f1'))
            >>> float(default.factor_irf(0)[0][0, 1]), bool(first.factor_irf(0)[0][0, 0] != 0.0)
            (0.0, True)
            >>> FactorAugmentedSVAR(res, order=("r", "z")).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: order must be a permutation of the variable ...
        """
        structural: StructuralResult = (
            self._structural
            if self._structural is not None
            else RecursiveSVAR(self._favar.factors, order=self._order).identify()
        )
        return FactorAugmentedSVARResult(favar=self._favar, structural=structural)
