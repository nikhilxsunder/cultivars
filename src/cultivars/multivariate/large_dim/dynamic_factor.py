# filepath: /src/cultivars/multivariate/large_dim/dynamic_factor.py
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
r"""The dynamic factor model: a large panel explained by a small VAR.

The premise (Stock and Watson; Forni, Hallin, Lippi and Reichlin) is that
a hundred macroeconomic series comove because a handful of common factors
drive them,

.. math::

   x_t = \Lambda F_t + e_t,
   \qquad
   F_t = \sum_{i=1}^{p} A_i F_{t-i} + u_t,

each standardized series a loading on :math:`r \ll N` factors plus an
idiosyncratic remainder, and the factors themselves a VAR. Estimation
here is the Stock-Watson two-step, deliberately: the first :math:`r`
principal components of the standardized panel estimate the factor space
consistently as both :math:`N` and :math:`T` grow, a least-squares VAR on
the scores captures the common dynamics, and the whole procedure is one
singular-value decomposition and one regression -- transparent, fast at
any :math:`N`, and free of the convergence pathologies an EM pass can
hide. The factor VAR is a full
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`,
so everything a VAR result can do -- forecast, impulse responses,
variance decompositions, spillover tables -- runs on the factor system
directly, and the panel's own forecast is the factor forecast mapped
back through the loadings with the idiosyncratic components at their
zero mean.

Two commitments shape the surface. First, the factor count is a
modelling choice the model makes only by a stated criterion: left
unspecified it minimizes Bai and Ng's :math:`IC_{p2}`, the standard
consistent selector, and the criterion values are kept on the result so
the choice can be audited rather than trusted; when the caller fixes the
count the audit trail is empty, because nothing was decided. Second, the
two identification facts are stated rather than hidden. Factors are
identified only up to rotation -- principal components pin the space,
not the basis -- so individual factors carry no structural names and the
summary says so; anything worth interpreting is read from the common
component, the loading patterns or the panel forecast, all of which are
rotation-invariant. And the two-step ignores estimation error in the
factors when fitting their VAR, which is asymptotically negligible for
large panels (Bai and Ng, 2006) and is stated on the summary for small
ones.

Layout. :class:`DFM` validates the panel through
``validate_endog_matrix`` from ``_core``, standardizes it, selects the
count with its own ``_bai_ng`` over ``principal_components`` from
``_core``, and fits the scores with
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
under ``trend="n"``. :class:`DFMResult` carries the loadings, the
standardization and the remainders, and takes the summary from
``_SummaryMixin``. The factors augmented with observed variables are
:mod:`~cultivars.multivariate.large_dim.factor_augmented`; the factor
model whose variances move is
:mod:`~cultivars.multivariate.large_dim.factor_volatility`; the
connectedness of the factor system is
:mod:`~cultivars.multivariate.large_dim.spillover`.

References:
    Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
    components from a large number of predictors. *Journal of the
    American Statistical Association*, 97(460), 1167-1179.

    Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The
    generalized dynamic-factor model: Identification and estimation.
    *Review of Economics and Statistics*, 82(4), 540-554.

    Bai, J., & Ng, S. (2002). Determining the number of factors in
    approximate factor models. *Econometrica*, 70(1), 191-221.

    Bai, J., & Ng, S. (2006). Confidence intervals for diffusion index
    forecasts and inference for factor-augmented regressions.
    *Econometrica*, 74(4), 1133-1150.

Example:
    Forty series driven by two factors: the count is selected, the
    common component is rotation-invariant although the factor labels
    are not, and the factor VAR feeds the spillover table directly:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.spillover import Spillover
    >>> rng = np.random.default_rng(0)
    >>> f = np.zeros((300, 2))
    >>> for t in range(1, 300):
    ...     f[t] = np.array([[0.6, 0.2], [0.0, 0.5]]) @ f[t - 1] + rng.standard_normal(2)
    >>> lam = rng.standard_normal((40, 2))
    >>> panel = f @ lam.T + rng.standard_normal((300, 40))
    >>> res = DFM(panel, order=1).fit()
    >>> res.n_factors, res.factors.is_stable
    (2, True)
    >>> theta = 0.7
    >>> rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    >>> rotated = (res.factors.endog @ rotation) @ (res.loadings @ rotation).T
    >>> bool(np.allclose(rotated * res.scales + res.means, res.common_component))
    True
    >>> table = Spillover(res.factors, horizon=10).compute()
    >>> table.names, bool(0.0 < table.total < 50.0)
    (('f1', 'f2'), True)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...engine._core import SummaryTable, principal_components, validate_endog_matrix
from ...engine._internals import _SummaryMixin
from ...exceptions import DimensionError, SpecificationError
from ..reduced_form.vector_autoregression import VAR, VARResult

__all__ = ["DFM", "DFMResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DFMResult(_SummaryMixin):
    r"""A fitted dynamic factor model: loadings, a factor VAR, and the split.

    The estimates of the two-step model on the standardized panel
    :math:`x_t = (y_t - \bar y) / s`,

    .. math::

       x_t = \Lambda F_t + e_t,
       \qquad
       F_t = \sum_{i=1}^{p} A_i F_{t-i} + u_t,

    with :math:`\Lambda` the first :math:`r` orthonormal right singular
    vectors of the standardized panel, :math:`F_t` its projection onto
    them, and the VAR fitted to :math:`F_t` by least squares without an
    intercept (the scores have mean zero by construction). The factor VAR
    is a complete fitted result of its own -- forecast it, decompose it,
    feed it to :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
    -- while this result owns the mapping between the panel and the
    factor space: the loadings, the standardization that was undone, the
    idiosyncratic remainders and the share of variance the factors carry.

    Note:
        The factor space is identified, the basis is not: any rotation of
        :math:`(\Lambda, F_t)` by an orthogonal :math:`r \times r` matrix
        leaves ``common_component``, ``r_squared`` and ``forecast``
        unchanged, so individual factors carry no structural names.
        Because the loadings are orthonormal and the standardized columns
        have unit variance, the mean of ``r_squared`` over the series
        equals ``variance_shares.sum()`` exactly. A constant column is
        standardized with scale 1 (not 0), contributes nothing to the
        factors, and reports an ``r_squared`` of 1 because its remainder
        is identically zero; read such a value as "nothing to explain",
        not "fully explained".

    Attributes:
        panel: The observed ``(nobs, n_series)`` panel.
        series_names: One label per series.
        factors: The fitted VAR on the estimated factors, a full
            :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
            with variables ``f1 ... fr``.
        loadings: ``(n_series, r)`` orthonormal loadings on the
            *standardized* panel; the factor space is identified, the basis
            is not.
        means: ``(n_series,)`` series means removed before extraction.
        scales: ``(n_series,)`` series standard deviations divided out.
        variance_shares: ``(r,)`` share of the standardized panel's variance
            each component explains.
        idiosyncratic: ``(nobs, n_series)`` standardized-panel remainders
            after the common component.
        criterion_values: Bai-Ng ``ICp2`` at each candidate count -- empty
            when the caller fixed the count, so the audit trail matches what
            was actually decided.
        n_factors: The factor count estimated with.

    See Also:
        * :class:`DFM` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR`
          -- the factors joined to observed variables in one VAR.
        * :class:`~cultivars.multivariate.large_dim.factor_volatility.FactorSV`
          -- the factor model with stochastic volatility.

    References:
        Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
        components from a large number of predictors. *Journal of the
        American Statistical Association*, 97(460), 1167-1179.

        Bai, J., & Ng, S. (2002). Determining the number of factors in
        approximate factor models. *Econometrica*, 70(1), 191-221.

    Example:
        Two persistent factors loaded onto thirty noisy series; the count
        is recovered, the common component tracks the truth, and the
        factor VAR is a full result:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> f = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal(2)
        >>> lam = rng.standard_normal((30, 2))
        >>> panel = f @ lam.T + 1.5 * rng.standard_normal((200, 30))
        >>> res = DFM(panel, order=1).fit()
        >>> res.n_factors, res.n_series, res.nobs, res.loadings.shape
        (2, 30, 200, (30, 2))
        >>> bool(np.allclose(res.loadings.T @ res.loadings, np.eye(2)))
        True
        >>> truth = f @ lam.T
        >>> bool(np.corrcoef(res.common_component[:, 0], truth[:, 0])[0, 1] > 0.9)
        True
        >>> res.factors.names, res.factors.order, res.criterion_values.shape
        (('f1', 'f2'), 1, (8,))
        >>> bool(abs(np.mean(list(res.r_squared().values())) - res.variance_shares.sum()) < 1e-12)
        True
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The observed ``(nobs, n_series)`` panel. Kept out of the repr."""
    series_names: tuple[str, ...]
    """One label per series, in column order."""
    factors: VARResult = field(repr=False)
    """The fitted VAR on the factor scores, variables ``f1 ... fr``. Kept out of the repr."""
    loadings: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series, r)`` orthonormal loadings on the standardized panel. Kept out of the repr."""
    means: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` means removed before extraction. Kept out of the repr."""
    scales: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` standard deviations divided out; 1 for a constant column. Kept out."""
    variance_shares: npt.NDArray[np.float64] = field(repr=False)
    """``(r,)`` share of standardized variance per component. Kept out of the repr."""
    idiosyncratic: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, n_series)`` standardized remainders. Kept out of the repr."""
    criterion_values: npt.NDArray[np.float64] = field(repr=False)
    """Bai-Ng ``ICp2`` per candidate count, empty when fixed. Kept out of the repr."""
    n_factors: int
    """The factor count the model was estimated with."""

    @property
    def n_series(self) -> int:
        """Number of series in the panel.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 12))
            >>> DFM(panel, order=1, n_factors=2).fit().n_series
            12
        """
        return len(self.series_names)

    @property
    def nobs(self) -> int:
        """Panel length.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 12))
            >>> DFM(panel, order=1, n_factors=2).fit().nobs
            100
        """
        return int(self.panel.shape[0])

    @property
    def common_component(self) -> npt.NDArray[np.float64]:
        r"""The factors' reconstruction of the panel, in original units.

        :math:`\hat y_t = s \odot (\Lambda F_t) + \bar y`, shape
        ``(nobs, n_series)``; invariant to the rotation of the factor
        basis. Together with the de-standardized ``idiosyncratic`` it
        reproduces the panel exactly.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal((150, 1))
            >>> panel = f @ rng.standard_normal((1, 8)) + 0.5 * rng.standard_normal((150, 8))
            >>> res = DFM(panel, order=1, n_factors=1).fit()
            >>> common = res.common_component
            >>> common.shape
            (150, 8)
            >>> rebuilt = common + res.idiosyncratic * res.scales
            >>> bool(np.allclose(rebuilt, panel))
            True
        """
        scores = np.asarray(self.factors.endog, dtype=np.float64)
        return scores @ self.loadings.T * self.scales + self.means

    def r_squared(self) -> dict[str, float]:
        r"""Share of each series' variance the common component explains.

        On the standardized scale each column has unit variance, so the
        share is :math:`1 - \operatorname{Var}(e_i)`; its mean across
        series is ``variance_shares.sum()``.

        Returns:
            A mapping from series name to its common-component ``R^2`` on
            the standardized scale.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal((150, 1))
            >>> lam = np.array([[2.0, 1.0, 0.0]])
            >>> panel = f @ lam + 0.5 * rng.standard_normal((150, 3))
            >>> fit = DFM(panel, order=1, n_factors=1, series_names=("a", "b", "c")).fit()
            >>> fit = fit.r_squared()
            >>> list(fit), bool(fit["a"] > fit["b"] > fit["c"])
            (['a', 'b', 'c'], True)
            >>> bool(fit["a"] > 0.9), bool(fit["c"] < 0.1)
            (True, True)
        """
        explained = 1.0 - self.idiosyncratic.var(axis=0)
        return {name: float(explained[i]) for i, name in enumerate(self.series_names)}

    def forecast(self, steps: int = 8) -> npt.NDArray[np.float64]:
        """Forecast the whole panel through the factor VAR.

        The factors are forecast by their own fitted system and mapped back
        through the loadings; the idiosyncratic components are forecast at
        their zero mean, which is the model's own claim about them. The
        forecasts therefore decay toward the series means at the factor
        VAR's rate and carry no idiosyncratic persistence.

        Args:
            steps: Horizons ahead, at least one.

        Returns:
            An array of shape ``(steps, n_series)`` in original units.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = np.zeros((200, 1))
            >>> for t in range(1, 200):
            ...     f[t] = 0.8 * f[t - 1] + rng.standard_normal()
            >>> panel = 5.0 + f @ rng.standard_normal((1, 6)) + 0.5 * rng.standard_normal((200, 6))
            >>> res = DFM(panel, order=1, n_factors=1).fit()
            >>> path = res.forecast(12)
            >>> path.shape
            (12, 6)
            >>> gap = np.abs(path - res.means).max(axis=1)
            >>> bool(gap[-1] < gap[0])
            True
            >>> res.forecast(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
        """
        scores = self.factors.forecast(steps)
        return scores @ self.loadings.T * self.scales + self.means

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            The ten series with the highest common ``R^2``, one row each;
            the metadata carries the specification, the panel size and
            the total common variance; the notes state the factor count's
            origin, the rotation caveat, the two-step caveat and that the
            factor VAR is a full result.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 12))
            >>> table = DFM(panel, order=1, n_factors=2).fit()._summary_table()
            >>> table.title, table.metadata[0], len(table.rows), len(table.notes)
            ('DFM(2) Results', ('Model', 'DFM(r=2, p=1)'), 10, 5)
            >>> table.notes[1]
            'The factor count was fixed by the caller.'
        """
        fit = self.r_squared()
        ordered = sorted(fit.items(), key=lambda item: item[1], reverse=True)
        rows = tuple((name, f"{value:.3f}") for name, value in ordered[:10])
        notes = [
            f"{self.n_factors} factor(s) explain "
            f"{100.0 * float(self.variance_shares.sum()):.1f}% of the "
            "standardized panel's variance.",
            (
                "The factor count minimized Bai-Ng ICp2; criterion_values keeps the audited path."
                if self.criterion_values.size
                else "The factor count was fixed by the caller."
            ),
            "Factors are identified up to rotation: the common component "
            "and loading patterns are interpretable, individual factor "
            "labels are not.",
            "Two-step estimation ignores factor-estimation error in the "
            "factor VAR; negligible for large panels (Bai-Ng 2006), stated "
            "for small ones.",
            "The factor VAR is a full VARResult on `factors`; forecast, "
            "IRFs, and connectedness run on it directly.",
        ]
        return SummaryTable(
            title=f"DFM({self.n_factors}) Results",
            metadata=(
                ("Model", f"DFM(r={self.n_factors}, p={self.factors.order})"),
                ("Series", f"{self.n_series}"),
                ("Observations", f"{self.nobs}"),
                ("Common variance", f"{100.0 * float(self.variance_shares.sum()):.1f}%"),
            ),
            columns=("series (top 10 by fit)", "common R^2"),
            rows=rows,
            notes=tuple(notes),
        )


class DFM:
    r"""Dynamic factor model, Stock-Watson two-step.

    Principal components of the standardized panel estimate the factor
    space; a VAR on the scores estimates the common dynamics. Left
    unspecified, the factor count minimizes Bai and Ng's ``ICp2``,

    .. math::

       IC_{p2}(r) = \ln V(r) + r\,\frac{N + T}{N T}\,\ln\min(N, T),

    with :math:`V(r)` the mean squared residual of the :math:`r`-component
    approximation, over ``1 .. max_factors``; the criterion values are
    kept on the result so the choice can be audited. The procedure is
    transparent and fast at any width: one singular-value decomposition
    and one least-squares VAR.

    Attributes:
        _panel: The validated ``(nobs, n_series)`` panel as a float array.
        _order: The autoregressive order of the factor VAR.
        _n_factors: The fixed factor count, or ``None`` to select.
        _max_factors: The largest candidate count, capped at what the
            panel can support.
        _series_names: One label per column.

    Args:
        panel: The observed ``(nobs, n_series)`` panel; wide is welcome.
        order: Autoregressive order of the factor VAR, at least 0.
        n_factors: Factor count, or ``None`` to minimize Bai-Ng ``ICp2``
            over ``1 .. max_factors``.
        max_factors: Largest candidate count for the criterion search,
            at least 1; silently capped at ``min(nobs - 1, n_series)``.
        series_names: One label per series. Defaults to ``x1 ... xN``.

    Raises:
        SpecificationError: If ``n_factors`` or ``max_factors`` is not a
            positive integer, or ``series_names`` does not have one entry
            per column. A negative ``order`` is rejected by the factor
            VAR at ``fit``.
        DimensionError: If the panel is not a time-down-the-rows matrix
            or ``n_factors`` exceeds ``min(nobs - 1, n_series)``.

    See Also:
        * :class:`DFMResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the model fitted to the factor scores.
        * :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR`
          -- the same factors augmented with observed variables.

    References:
        Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
        components from a large number of predictors. *Journal of the
        American Statistical Association*, 97(460), 1167-1179.

        Bai, J., & Ng, S. (2002). Determining the number of factors in
        approximate factor models. *Econometrica*, 70(1), 191-221.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> f = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal(2)
        >>> lam = rng.standard_normal((30, 2))
        >>> panel = f @ lam.T + 1.5 * rng.standard_normal((200, 30))
        >>> res = DFM(panel, order=1).fit()
        >>> res.n_factors
        2
        >>> res.factors.coefficients.shape
        (1, 2, 2)
        >>> fixed = DFM(panel, order=2, n_factors=3).fit()
        >>> fixed.n_factors, fixed.criterion_values.shape, fixed.factors.order
        (3, (0,), 2)
        >>> DFM(panel, order=1, n_factors=31)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.DimensionError: n_factors (31) exceeds what a (200, 30) panel ...
    """

    __slots__ = ("_max_factors", "_n_factors", "_order", "_panel", "_series_names")

    def __init__(
        self,
        panel: npt.ArrayLike,
        *,
        order: int,
        n_factors: int | None = None,
        max_factors: int = 8,
        series_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel and the factor-count request.

        Args:
            panel: The observed ``(nobs, n_series)`` panel.
            order: Autoregressive order of the factor VAR.
            n_factors: Factor count, or ``None`` to select by criterion.
            max_factors: Largest candidate count for the search.
            series_names: One label per series, or ``None`` for defaults.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 5))
            >>> model = DFM(panel, order=1, max_factors=8)
            >>> model._n_factors, model._max_factors, model._series_names[-1]
            (None, 5, 'x5')
        """
        self._panel = validate_endog_matrix(panel)
        nobs, n_series = self._panel.shape
        ceiling = min(nobs - 1, n_series)
        if n_factors is not None:
            if int(n_factors) != n_factors or n_factors < 1:
                raise SpecificationError(f"n_factors must be an integer >= 1; got {n_factors!r}.")
            if n_factors > ceiling:
                raise DimensionError(
                    f"n_factors ({n_factors}) exceeds what a ({nobs}, "
                    f"{n_series}) panel can support."
                )
        if int(max_factors) != max_factors or max_factors < 1:
            raise SpecificationError(f"max_factors must be an integer >= 1; got {max_factors!r}.")
        self._n_factors = None if n_factors is None else int(n_factors)
        self._max_factors = min(int(max_factors), ceiling)
        self._order = int(order)
        if series_names is None:
            self._series_names = tuple(f"x{i + 1}" for i in range(n_series))
        else:
            resolved = tuple(str(name) for name in series_names)
            if len(resolved) != n_series:
                raise SpecificationError(
                    f"series_names must have one entry per series ({n_series}); "
                    f"got {len(resolved)}."
                )
            self._series_names = resolved

    @staticmethod
    def _bai_ng(
        standardized: npt.NDArray[np.float64], max_factors: int
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Bai-Ng ``ICp2`` over candidate counts, and its argmin.

        ``ICp2(r) = ln V(r) + r ((N + T) / (N T)) ln(min(N, T))`` with
        ``V(r)`` the mean squared residual of the ``r``-component
        approximation -- the paper's preferred penalty, consistent as both
        dimensions grow. One finite-sample honesty note: when the common
        component explains nearly all of the panel, the log criterion
        measures each spurious eigenvalue against a tiny remainder and
        overselects -- a known property of the criterion, not of this
        implementation. If the reported count looks generous next to the
        variance shares, state ``n_factors`` yourself.

        Args:
            standardized: The ``(nobs, n_series)`` panel with zero column
                means and unit column variances.
            max_factors: The largest candidate count, at most
                ``min(nobs, n_series)``.

        Returns:
            ``(count, values)``: the minimizing count, from 1, and the
            criterion at each candidate ``1 .. max_factors``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = np.zeros((200, 2))
            >>> for t in range(1, 200):
            ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal(2)
            >>> panel = f @ rng.standard_normal((30, 2)).T + 1.5 * rng.standard_normal((200, 30))
            >>> z = (panel - panel.mean(axis=0)) / panel.std(axis=0)
            >>> count, values = DFM._bai_ng(z, 6)
            >>> count, values.shape, int(np.argmin(values)) + 1
            (2, (6,), 2)
        """
        nobs, n_series = standardized.shape
        scores, loadings, _ = principal_components(standardized, max_factors)
        penalty = (n_series + nobs) / (n_series * nobs) * np.log(min(n_series, nobs))
        values = np.empty(max_factors)
        for r in range(1, max_factors + 1):
            resid = standardized - scores[:, :r] @ loadings[:, :r].T
            values[r - 1] = float(np.log(np.mean(resid**2)) + r * penalty)
        return int(np.argmin(values)) + 1, values

    def fit(self) -> DFMResult:
        """Extract the factors, count them if unasked, and fit their VAR.

        Standardizes each column by its mean and population standard
        deviation (a constant column keeps scale 1), selects or accepts
        the count, takes the principal components, and fits a VAR with no
        deterministic terms to the scores under the names ``f1 ... fr``.

        Returns:
            The fitted :class:`DFMResult`.

        Raises:
            SpecificationError: If ``order`` is negative, from the factor
                VAR.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> panel = 3.0 + rng.standard_normal((120, 6)) @ rng.standard_normal((6, 6))
            >>> res = DFM(panel, order=1, n_factors=2).fit()
            >>> bool(np.allclose(res.means, panel.mean(axis=0)))
            True
            >>> bool(np.allclose(res.scales, panel.std(axis=0)))
            True
            >>> bool(abs(res.factors.endog.mean()) < 1e-12)
            True
        """
        means = self._panel.mean(axis=0)
        scales = self._panel.std(axis=0, ddof=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (self._panel - means) / scales
        if self._n_factors is None:
            count, criterion = self._bai_ng(standardized, self._max_factors)
        else:
            count, criterion = self._n_factors, np.zeros(0)
        scores, loadings, shares = principal_components(standardized, count)
        factor_result = VAR(
            scores,
            order=self._order,
            trend="n",
            names=tuple(f"f{i + 1}" for i in range(count)),
        ).fit()
        return DFMResult(
            panel=self._panel,
            series_names=self._series_names,
            factors=factor_result,
            loadings=loadings,
            means=means,
            scales=scales,
            variance_shares=shares,
            idiosyncratic=standardized - scores @ loadings.T,
            criterion_values=criterion,
            n_factors=count,
        )
