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
r"""The factor-augmented VAR: a few shocks, answered by hundreds of series.

A small VAR cannot say what a policy shock does to the hundreds of series
a policymaker actually watches, and a VAR on hundreds of series cannot be
estimated. The factor-augmented VAR threads the needle (Bernanke, Boivin
and Eliasz, 2005): a wide informational panel is summarized by a handful
of principal-component factors, a VAR runs on those factors augmented
with the observed variables of interest, and an observation equation
maps everything the small system produces back onto every series,

.. math::

   \begin{pmatrix} F_t \\ Y_t \end{pmatrix}
   = \sum_{i=1}^{p} A_i \begin{pmatrix} F_{t-i} \\ Y_{t-i} \end{pmatrix}
   + u_t,
   \qquad
   x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t,

so a shock identified in the :math:`r + m` dimensional system has a
response on all :math:`N` panel series through :math:`\Lambda`.

The design is composition, on the pattern the functional VAR set: the
factor dynamics *are* a
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
on the augmented block, riding on the result as ``factors`` with its
entire reduced-form surface intact. Because that result is a closed
system, the entire structural layer applies to it unchanged --
recursive, sign, narrative, proxy, heteroskedasticity, any of them -- and
:class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
translates whatever was identified into responses of all :math:`N` panel
series. That translation is rotation-invariant: principal components
recover the factor space only up to rotation, but loadings and factors
rotate together, so panel responses to an identified shock do not depend
on the arbitrary factor basis.

Two commitments shape the surface. First, the estimation subtlety that
is load-bearing for policy analysis is handled the way the original
paper handles it, and its absence is disclosed. Principal components of
the *full* panel absorb the observed variables' contemporaneous
influence, which contaminates a recursive identification that orders
those variables last. Declaring the ``slow`` series -- those that do not
react within the period -- triggers the Bernanke-Boivin-Eliasz cleaning,
:math:`\hat F_t = C_t - \hat b_Y' Y_t` with :math:`\hat b_Y` from the
regression of the full-panel components on the slow-panel components and
:math:`Y_t`; skipping the declaration puts a warning on the summary
rather than papering over it. Second, nothing structural happens here.
The result is reduced-form -- a VAR, loadings and a fit -- and every
identifying assumption is made explicitly, afterwards, in
:mod:`cultivars.multivariate.structural`, where it is named and
auditable.

Layout. :class:`FAVAR` validates the panel through
``_validate_wide_panel`` and the observed block through
``validate_exog_matrix`` from ``_core``, standardizes, extracts with
``principal_components``, cleans by least squares, and fits the scores
with :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`.
:class:`FAVARResult` carries the loadings, the scores and the
standardization, and takes the summary from ``_SummaryMixin``. The
factor model without an observed block is
:mod:`~cultivars.multivariate.large_dim.dynamic_factor`; the
identification step is
:mod:`~cultivars.multivariate.structural.factor_augmented`.

References:
    Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
    effects of monetary policy: A factor-augmented vector autoregressive
    (FAVAR) approach. *Quarterly Journal of Economics*, 120(1), 387-422.

    Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
    components from a large number of predictors. *Journal of the
    American Statistical Association*, 97(460), 1167-1179.

Example:
    A panel whose fast half reacts to the policy shock within the period
    and whose slow half does not. With the slow series declared, a
    recursive identification ordering the rate last recovers that
    structure in the impact responses of the panel:

    >>> import numpy as np
    >>> from cultivars.multivariate.structural.factor_augmented import FactorAugmentedSVAR
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> factor = np.zeros(n)
    >>> for t in range(1, n):
    ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
    >>> shock = rng.standard_normal(n)
    >>> policy = 0.5 * factor + shock
    >>> lam = rng.uniform(0.5, 1.5, 30)
    >>> slow_half = np.outer(factor, lam[:15]) + 0.5 * rng.standard_normal((n, 15))
    >>> fast_half = np.outer(factor, lam[15:]) + 0.5 * rng.standard_normal((n, 15))
    >>> fast_half += 0.8 * np.outer(shock, rng.uniform(0.5, 1.5, 15))
    >>> panel = np.column_stack([slow_half, fast_half])
    >>> names = tuple(f"s{i}" for i in range(15)) + tuple(f"q{i}" for i in range(15))
    >>> res = FAVAR(
    ...     panel, policy, n_factors=1, order=1, panel_names=names,
    ...     slow=names[:15], observed_names=("rate",),
    ... ).fit()
    >>> res.cleaned, res.factors.names
    (True, ('f1', 'rate'))
    >>> svar = FactorAugmentedSVAR(res).identify()
    >>> svar.shock_names, svar.irf(8).shape
    (('f1', 'rate'), (9, 30, 2))
    >>> impact = np.abs(svar.impact[:, 1])
    >>> bool(impact[15:].mean() > 10 * impact[:15].mean())
    True
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
    principal_components,
    validate_exog_matrix,
)
from ..._internals import _SummaryMixin
from ...exceptions import SpecificationError
from ..reduced_form.vector_autoregression import VAR, VARResult

__all__ = ["FAVAR", "FAVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FAVARResult(_SummaryMixin):
    r"""A fitted factor-augmented vector autoregression.

    The estimates of

    .. math::

       \begin{pmatrix} F_t \\ Y_t \end{pmatrix}
       = \sum_{i=1}^{p} A_i \begin{pmatrix} F_{t-i} \\ Y_{t-i} \end{pmatrix}
       + u_t,
       \qquad
       x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t,

    where :math:`F_t` are the principal-component factors of the
    standardized panel :math:`x_t` (cleaned of :math:`Y_t`'s
    contemporaneous influence when slow series were declared),
    :math:`Y_t` is the observed block, and the observation equation is
    fitted by least squares one panel series at a time. Composition,
    stated plainly: :attr:`factors` is a complete
    :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
    on the augmented block, and every factor-level question --
    coefficients, diagnostics, stability, reduced-form impulse responses
    -- is answered there. It is also a closed system, so every
    identification model in :mod:`cultivars.multivariate.structural`
    accepts it directly, and
    :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
    maps an identified factor-space shock onto responses of every series
    through ``loadings``. This object owns what the factor result cannot
    know: the panel, the loadings and the standardization that was
    undone.

    Note:
        ``explained_variance`` is the principal components' share
        *before* cleaning, since cleaning moves the factors off the
        component basis. ``r2`` and ``common_component`` are unaffected
        by cleaning: the cleaned factors span, together with the observed
        block, the same column space as the raw ones, so the observation
        equation fits identically. The factor labels ``f1 ... fr`` are
        a basis, not names -- principal components are identified up to
        rotation, and ``loadings`` rotate with them, so panel responses to
        an identified shock do not depend on the basis while a
        factor-by-factor reading does.

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

    See Also:
        * :class:`FAVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
          -- identification on ``factors`` mapped onto the panel.
        * :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFMResult`
          -- the factor model without an observed block.

    References:
        Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
        effects of monetary policy: A factor-augmented vector
        autoregressive (FAVAR) approach. *Quarterly Journal of
        Economics*, 120(1), 387-422.

    Example:
        One factor, thirty series, a policy rate that loads on the
        factor; the augmented VAR has two variables and the observation
        equation explains most of the panel:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> factor = np.zeros(300)
        >>> for t in range(1, 300):
        ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
        >>> panel = np.outer(factor, rng.uniform(0.5, 1.5, 30))
        >>> panel += 0.3 * rng.standard_normal((300, 30))
        >>> policy = 0.4 * factor + rng.standard_normal(300)
        >>> res = FAVAR(panel, policy[:, None], n_factors=1, order=1).fit()
        >>> res.n_series, res.n_factors, res.factors.names, res.cleaned
        (30, 1, ('f1', 'y1'), False)
        >>> res.loadings.shape, res.scores.shape, res.explained_variance.shape
        ((30, 2), (300, 1), (1,))
        >>> bool(res.r2.mean() > 0.9), bool(abs(np.corrcoef(res.scores[:, 0], factor)[0, 1]) > 0.95)
        (True, True)
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The informational panel as supplied, ``(nobs, n_series)``. Kept out of the repr."""
    observed: npt.NDArray[np.float64] = field(repr=False)
    """The observed block as supplied, ``(nobs, m)``. Kept out of the repr."""
    factors: VARResult = field(repr=False)
    """The fitted VAR on ``[F, Y]``, variables ``f1 ... fr`` then the observed names."""
    loadings: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series, r + m)`` observation-equation coefficients. Kept out of the repr."""
    scores: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, r)`` estimated factors, cleaned when ``slow`` was declared. Kept out."""
    panel_names: tuple[str, ...]
    """One label per panel series, in column order."""
    explained_variance: npt.NDArray[np.float64] = field(repr=False)
    """``(r,)`` variance share per component, before cleaning. Kept out of the repr."""
    r2: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` observation-equation fit per series. Kept out of the repr."""
    cleaned: bool
    """Whether the Bernanke-Boivin-Eliasz slow-variable cleaning ran."""
    _scales: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` standard deviations divided out of the panel. Kept out of the repr."""

    @property
    def n_series(self) -> int:
        """Number of panel series.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = FAVAR(
            ...     rng.standard_normal((120, 8)), rng.standard_normal((120, 2)),
            ...     n_factors=2, order=1,
            ... ).fit()
            >>> res.n_series, res.n_factors, res.factors.k_endog
            (8, 2, 4)
        """
        return len(self.panel_names)

    @property
    def n_factors(self) -> int:
        """Number of estimated factors.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = FAVAR(
            ...     rng.standard_normal((120, 8)), rng.standard_normal((120, 1)),
            ...     n_factors=3, order=1,
            ... ).fit()
            >>> res.n_factors, res.scores.shape, res.explained_variance.shape
            (3, (120, 3), (3,))
        """
        return int(self.scores.shape[1])

    def common_component(self) -> npt.NDArray[np.float64]:
        r"""The panel as the factors and observed block see it, in original units.

        :math:`\hat x_t = s \odot (\Lambda^F F_t + \Lambda^Y Y_t) + \bar x`.

        Returns:
            An ``(nobs, n_series)`` array; the gap between this and
            :attr:`panel` is the idiosyncratic component, and each series'
            :attr:`r2` says how much of it there is.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> factor = np.zeros(300)
            >>> for t in range(1, 300):
            ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
            >>> panel = np.outer(factor, rng.uniform(0.5, 1.5, 30))
            >>> panel += 0.3 * rng.standard_normal((300, 30))
            >>> policy = 0.4 * factor + rng.standard_normal(300)
            >>> res = FAVAR(panel, policy[:, None], n_factors=1, order=1).fit()
            >>> common = res.common_component()
            >>> common.shape
            (300, 30)
            >>> total = ((panel - panel.mean(axis=0)) ** 2).sum(axis=0)
            >>> fit = 1.0 - ((panel - common) ** 2).sum(axis=0) / total
            >>> bool(np.allclose(fit, res.r2))
            True
        """
        augmented = np.column_stack([self.scores, self.observed])
        centered = augmented @ self.loadings.T * self._scales
        return centered + self.panel.mean(axis=0)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per factor with its pre-cleaning variance share; the
            metadata carries the block sizes, the order, whether the
            cleaning ran, the mean observation fit and the factor VAR's
            log-likelihood; the notes state the stability verdict, the
            structural hook, the rotation caveat and -- when no slow
            series were declared -- the contamination warning.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> panel = rng.standard_normal((150, 10))
            >>> names = tuple(f"s{i}" for i in range(10))
            >>> raw = FAVAR(panel, rng.standard_normal((150, 1)), n_factors=1, order=1).fit()
            >>> table = raw._summary_table()
            >>> table.title, table.columns, len(table.rows), len(table.notes)
            ('FAVAR(1) Results', ('factor', 'variance share'), 1, 4)
            >>> table.metadata[5]
            ('Slow-variable cleaning', 'no')
            >>> clean = FAVAR(
            ...     panel, rng.standard_normal((150, 1)), n_factors=1, order=1,
            ...     panel_names=names, slow=names[:5],
            ... ).fit()
            >>> len(clean._summary_table().notes), clean._summary_table().metadata[5]
            (3, ('Slow-variable cleaning', 'yes'))
        """
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
    r"""Factor-augmented VAR over a wide informational panel, BBE (2005).

    Extract, clean, augment, fit. Principal components of the standardized
    panel estimate the factors; when ``slow`` series are declared, the
    observed block's contemporaneous influence is regressed out of them
    against the slow-panel components,

    .. math::

       \hat F_t = C_t - \hat b_Y' Y_t,
       \qquad
       C_t = \hat b_S' C^{\text{slow}}_t + \hat b_Y' Y_t + \text{error},

    with :math:`C_t` the full-panel components and :math:`C^{\text{slow}}_t`
    the components of the slow series alone, which is what makes a
    recursive policy identification on the result defensible; the
    cleaned factors and the observed block then form a small VAR
    estimated by composition; and the observation equation is fitted by
    least squares, one panel series at a time.

    Attributes:
        _panel: The validated ``(nobs, n_series)`` panel as a float array.
        _observed: The validated ``(nobs, m)`` observed block.
        _n_factors: The number of principal-component factors.
        _order: The autoregressive order of the augmented VAR.
        _slow: Column positions of the slow series, or ``None``.
        _trend: The deterministic terms of the augmented VAR.
        _observed_names: One label per observed variable.
        _panel_names: One label per panel series.

    Args:
        panel: The ``(nobs, n_series)`` informational panel. Standardized
            internally; responses are reported back in original units.
        observed: The ``(nobs, m)`` observed block -- the variables whose
            shocks are of interest, a policy rate being the canonical case.
            A one-dimensional array is read as one variable.
        n_factors: Number of principal-component factors to extract, at
            least 1 and at most ``min(nobs - 1, n_series)``.
        order: Autoregressive order of the augmented VAR, at least 0.
        slow: Names of the panel series that do not respond to the observed
            block within the period, for the Bernanke-Boivin-Eliasz cleaning;
            at least ``n_factors + 1`` of them. ``None`` skips the cleaning,
            and the summary says so.
        trend: Deterministic terms of the augmented VAR, ``"n"``, ``"c"``
            or ``"ct"``.
        observed_names: Labels for the observed block. Defaults to
            ``y1 ... ym``.
        panel_names: Labels for the panel series. Defaults to ``x1 ... xN``.

    Raises:
        SpecificationError: If ``n_factors`` is not a positive integer or
            exceeds what the panel supports, a label set has the wrong
            length or repeats, a slow name is unknown, too few slow series
            are declared, or ``order`` or ``trend`` is rejected by the
            augmented VAR at ``fit``.
        DimensionError: If the panel is not a time-down-the-rows matrix or
            the observed block does not have one row per panel period.

    See Also:
        * :class:`FAVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
          -- the identification step on the result.
        * :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM` --
          the factor model without an observed block.

    References:
        Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
        effects of monetary policy: A factor-augmented vector
        autoregressive (FAVAR) approach. *Quarterly Journal of
        Economics*, 120(1), 387-422.

    Example:
        A panel whose fast half responds to the policy shock within the
        period. Without cleaning the factor absorbs the shock; declaring
        the slow half removes it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 400
        >>> factor = np.zeros(n)
        >>> for t in range(1, n):
        ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
        >>> shock = rng.standard_normal(n)
        >>> policy = 0.5 * factor + shock
        >>> lam = rng.uniform(0.5, 1.5, 30)
        >>> slow_half = np.outer(factor, lam[:15]) + 0.5 * rng.standard_normal((n, 15))
        >>> fast_half = np.outer(factor, lam[15:]) + 0.5 * rng.standard_normal((n, 15))
        >>> fast_half += 0.8 * np.outer(shock, rng.uniform(0.5, 1.5, 15))
        >>> panel = np.column_stack([slow_half, fast_half])
        >>> names = tuple(f"s{i}" for i in range(15)) + tuple(f"q{i}" for i in range(15))
        >>> raw = FAVAR(panel, policy, n_factors=1, order=1, panel_names=names).fit()
        >>> clean = FAVAR(
        ...     panel, policy, n_factors=1, order=1, panel_names=names, slow=names[:15]
        ... ).fit()
        >>> raw.cleaned, clean.cleaned, clean.factors.k_endog
        (False, True, 2)
        >>> leak = lambda res: abs(np.corrcoef(res.scores[:, 0], shock)[0, 1])
        >>> bool(leak(raw) > 0.3), bool(leak(clean) < 0.1)
        (True, True)
        >>> FAVAR(panel, policy, n_factors=1, order=1, slow=("x1",))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: the cleaning needs more slow series than ...
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
        """Validate the panel, the observed block, and the specification.

        Args:
            panel: The ``(nobs, n_series)`` informational panel.
            observed: The ``(nobs, m)`` observed block.
            n_factors: Number of principal-component factors.
            order: Autoregressive order of the augmented VAR.
            slow: Names of the slow panel series, or ``None``.
            trend: Deterministic terms of the augmented VAR.
            observed_names: Labels for the observed block, or ``None``.
            panel_names: Labels for the panel series, or ``None``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = FAVAR(
            ...     rng.standard_normal((120, 6)), rng.standard_normal((120, 2)),
            ...     n_factors=1, order=2, slow=("x1", "x3"),
            ... )
            >>> model._slow, model._observed_names, model._panel_names[-1]
            ((0, 2), ('y1', 'y2'), 'x6')
        """
        self._panel = _validate_wide_panel(panel, label="panel")
        nobs, n_series = self._panel.shape
        self._observed = validate_exog_matrix(observed, nobs=nobs, label="observed")
        if int(n_factors) != n_factors or n_factors < 1:
            raise SpecificationError(f"n_factors must be an integer >= 1; got {n_factors!r}.")
        if n_factors > min(nobs - 1, n_series):
            raise SpecificationError(
                f"n_factors ({n_factors}) exceeds what a ({nobs}, {n_series}) panel can support."
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
                    f"panel_names must be {n_series} unique labels; got {len(resolved)}."
                )
            self._panel_names = resolved
        m = self._observed.shape[1]
        if observed_names is None:
            self._observed_names = tuple(f"y{i + 1}" for i in range(m))
        else:
            resolved = tuple(str(name) for name in observed_names)
            if len(resolved) != m or len(set(resolved)) != m:
                raise SpecificationError(
                    f"observed_names must be {m} unique labels; got {len(resolved)}."
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
                        f"unknown slow series {label!r}; expected one of the panel names."
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
        """Principal-component scores and variance shares of a panel.

        Args:
            standardized: The ``(nobs, k)`` panel with zero column means and
                unit column variances.
            count: Number of components, at most ``min(nobs, k)``.

        Returns:
            ``(scores, shares)``: the ``(nobs, count)`` scores and the
            ``(count,)`` variance shares; the loadings are discarded
            because the observation equation refits them on the augmented
            block.

        Example:
            >>> import numpy as np
            >>> z = np.random.default_rng(0).standard_normal((100, 5))
            >>> z = (z - z.mean(axis=0)) / z.std(axis=0)
            >>> scores, shares = FAVAR._components(z, 2)
            >>> scores.shape, shares.shape, bool(shares[0] >= shares[1])
            ((100, 2), (2,), True)
        """
        scores, _, shares = principal_components(standardized, count)
        return scores, shares

    def fit(self) -> FAVARResult:
        """Extract the factors, clean them, and estimate the augmented VAR.

        Standardizes the panel by its means and population standard
        deviations (a constant column keeps scale 1), takes the
        components, regresses the observed block out of them against the
        slow-panel components when ``slow`` was declared, fits the VAR on
        ``[F, Y]`` under the names ``f1 ... fr`` and the observed labels,
        and fits the observation equation of every standardized series on
        the augmented block.

        Returns:
            The fitted result, its ``factors`` ready for any identification
            model.

        Raises:
            SpecificationError: If ``order`` or ``trend`` is rejected by
                the augmented VAR.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> factor = np.zeros(300)
            >>> for t in range(1, 300):
            ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
            >>> panel = np.outer(factor, rng.uniform(0.5, 1.5, 30))
            >>> panel += 0.3 * rng.standard_normal((300, 30))
            >>> policy = 0.4 * factor + rng.standard_normal(300)
            >>> res = FAVAR(
            ...     panel, policy[:, None], n_factors=1, order=1, observed_names=("rate",)
            ... ).fit()
            >>> res.factors.names, res.factors.order, res.factors.is_stable
            (('f1', 'rate'), 1, True)
            >>> bool(abs(res.scores.mean()) < 1e-12), bool(res.r2.min() > 0.5)
            (True, True)
        """
        means = self._panel.mean(axis=0)
        scales = self._panel.std(axis=0, ddof=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (self._panel - means) / scales

        scores, shares = self._components(standardized, self._n_factors)
        cleaned = False
        if self._slow is not None:
            slow_scores, _ = self._components(standardized[:, list(self._slow)], self._n_factors)
            design = np.column_stack([slow_scores, self._observed])
            coef: npt.NDArray[np.float64] = np.linalg.lstsq(design, scores, rcond=None)[0]
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

        loadings: npt.NDArray[np.float64] = np.linalg.lstsq(augmented, standardized, rcond=None)[
            0
        ].T
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
