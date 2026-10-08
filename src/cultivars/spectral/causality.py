# filepath: /src/cultivars/spectral/causality.py
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
r"""Geweke's causality by frequency: *when* one variable's history matters.

A time-domain Granger statement answers whether :math:`x`'s history
helps predict :math:`y`; Geweke's (1982) spectral decomposition answers
at which frequencies. In the pair's innovation representation
:math:`(H, \Sigma)`, with the instantaneous correlation rotated out of
:math:`x`'s innovation, :math:`y`'s spectrum splits at each frequency
into an intrinsic part and a part attributable to :math:`x`'s
innovations, and the measure is the log ratio of the whole to the
intrinsic,

.. math::

   f_{x \to y}(\omega) = \ln \frac{S_{yy}(\omega)}
   {H_{yy}(\omega)\, \tilde\Sigma_{yy}\, \overline{H_{yy}(\omega)}},
   \qquad
   F_{x \to y} = \frac{1}{\pi} \int_0^{\pi} f_{x \to y}(\omega)\, d\omega
   = \ln \frac{\sigma^2_{\text{restricted}}}{\sigma^2_{\text{full}}},

so the integral over the circle recovers the time-domain measure -- and
a flat causality curve and a business-cycle-peaked one can carry the
same headline Granger number while saying entirely different things
about the economics.

Two objects, one honesty split. :class:`SpectralCausality` computes the
*unconditional* pairwise measure for every ordered pair, purely as a
view of one fitted system: for a bivariate model the transfer function
and covariance are the model's own; for a larger one, each pair's
measure needs the innovation representation of that pair's *marginal*
process, which is generally VARMA and matches no lag stack -- so it is
recovered exactly from the pair's marginal spectral density by Wilson's
spectral factorization (the route of Dhamala, Rangarajan & Ding 2008),
still with nothing re-estimated. :class:`ConditionalSpectralCausality`
computes Geweke's (1984) conditional measure, which mathematically
requires a second, restricted model -- there is no view-only shortcut
-- so that model is an explicit argument the caller fits, never
something estimated silently inside a "view", and the two models'
compatibility is reported as a diagnostic rather than assumed.

The family caveat applies with force: these are statements about fitted
models, read as association across frequencies, not causal claims about
the world; and pairwise unconditional measures can reflect common
driving by a third variable, which is exactly what the conditional
measure is for.

Layout. :class:`SpectralCausality` produces a
:class:`SpectralCausalityResult`, an ``(n, k, k)`` array of every ordered
pair with ``pair``, ``band``, and ``integrated`` readings;
:class:`ConditionalSpectralCausality` produces a
:class:`ConditionalSpectralCausalityResult`, one cause against every
effect. The numerics live in ``_core``: ``frequency_grid`` lays out
``[0, pi]``, ``transfer_function`` evaluates :math:`\Psi(\omega)` from
a lag stack, ``spectral_matrix`` forms :math:`\Psi \Sigma \Psi^\ast /
2\pi`, ``_pairwise_measure`` computes the bivariate measure from an
innovation representation, and ``ClosedSystemResult`` is the protocol a
fitted result must satisfy; ``_spectral_factor`` in ``_internals`` is
Wilson's algorithm. The time-domain Granger test on the same relation
is ``granger_causality`` on the fitted result, and the spectral matrix
and coherence the measures decompose are in
:mod:`~cultivars.spectral.density`.

References:
    Geweke, J. (1982). Measurement of linear dependence and feedback
    between multiple time series. *Journal of the American Statistical
    Association*, 77(378), 304-313.

    Geweke, J. (1984). Measures of conditional linear dependence and
    feedback between time series. *Journal of the American Statistical
    Association*, 79(388), 907-915.

    Dhamala, M., Rangarajan, G., & Ding, M. (2008). Estimating Granger
    causality from Fourier and wavelet transforms of time series data.
    *Physical Review Letters*, 100(1), 018701.

    Ding, M., Chen, Y., & Bressler, S. L. (2006). Granger causality: Basic
    theory and application to neuroscience. In *Handbook of Time Series
    Analysis* (pp. 437-460). Wiley.

    Wilson, G. T. (1972). The factorization of matricial spectral
    densities. *SIAM Journal on Applied Mathematics*, 23(4), 420-426.

Example:
    A rate that feeds into output; the spectral measure integrates to
    the time-domain Granger measure, and the business-cycle band carries
    more of it than the short cycles:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((500, 2))
    >>> for t in range(1, 500):
    ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
    ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
    >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
    >>> causal = SpectralCausality(res, n_frequencies=64).compute()
    >>> causal.integrated().round(3)
    array([[0.   , 0.154],
           [0.001, 0.   ]])
    >>> cycle = causal.band("rate", "gdp", low_period=6, high_period=32)
    >>> short = causal.band("rate", "gdp", low_period=2, high_period=4)
    >>> round(cycle, 2), round(short, 2)
    (0.26, 0.08)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    ClosedSystemResult,
    SummaryTable,
    _pairwise_measure,
    frequency_grid,
    spectral_matrix,
    transfer_function,
)
from .._internals import _spectral_factor, _SummaryMixin
from ..exceptions import SpecificationError

__all__ = [
    "ConditionalSpectralCausality",
    "ConditionalSpectralCausalityResult",
    "SpectralCausality",
    "SpectralCausalityResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralCausalityResult(_SummaryMixin):
    r"""All ordered pairs' unconditional Geweke measures across frequency.

    For each ordered pair the effect's spectrum at frequency
    :math:`\omega` is split, in the pair's own innovation representation
    :math:`(H, \Sigma)` with the instantaneous correlation rotated out of
    the cause's innovation, into an intrinsic part and a part driven by
    the cause's innovations, and the measure is the log ratio of total to
    intrinsic,

    .. math::

       f_{j \to i}(\omega) = \ln \frac{S_{ii}(\omega)}
       {H_{ii}(\omega)\, \tilde\Sigma_{ii}\, \overline{H_{ii}(\omega)}}
       \;\ge\; 0,

    with :math:`S_{ii} = [H \Sigma H^\ast]_{ii}` the effect's spectrum and
    :math:`\tilde\Sigma_{ii}` its innovation variance after the rotation.
    Zero at a frequency says the cause's history contributes nothing to
    the effect's variance at that cycle length; the integral over
    :math:`[0, \pi]` divided by :math:`\pi` recovers Geweke's time-domain
    measure :math:`\ln(\sigma^2_{\text{restricted}} / \sigma^2_{\text{full}})`,
    so the curve is the Granger statement resolved by frequency.

    Attributes:
        names: Variable labels, indexing the measure's matrix axes.
        frequencies: The ``(n,)`` grid on ``[0, pi]``.
        measure: The ``(n, k, k)`` measure; entry ``[w, i, j]`` is the
            causality from variable ``j`` to variable ``i`` at frequency
            ``w``. The diagonal is zero by convention.

    Note:
        The axis order is ``[effect, cause]``, the convention of the
        Geweke literature and of :meth:`integrated`, so ``measure[:, i,
        j]`` reads "``j`` causes ``i``"; :meth:`pair` takes the arguments
        in the spoken order ``(cause, effect)`` and does the transposition.
        The measures are unconditional: a third variable driving both
        members of a pair shows up as causality between them, which is
        what :class:`ConditionalSpectralCausalityResult` removes.

    See Also:
        * :class:`SpectralCausality` -- the producer.
        * :class:`ConditionalSpectralCausalityResult` -- the same measure
          with the rest of the system conditioned out.
        * :class:`~cultivars.spectral.density.SpectralDensityResult` --
          the spectral matrix and coherence the measure decomposes.

    References:
        Geweke, J. (1982). Measurement of linear dependence and feedback
        between multiple time series. *Journal of the American Statistical
        Association*, 77(378), 304-313.

        Ding, M., Chen, Y., & Bressler, S. L. (2006). Granger causality:
        Basic theory and application to neuroscience. In *Handbook of Time
        Series Analysis* (pp. 437-460). Wiley.

    Example:
        A rate that feeds into output with no feedback; the integrated
        spectral measure matches the time-domain Granger measure from
        the restricted and full regressions:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((500, 2))
        >>> for t in range(1, 500):
        ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
        ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
        >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
        >>> causal = SpectralCausality(res, n_frequencies=64).compute()
        >>> causal.measure.shape, causal.k_endog
        ((64, 2, 2), 2)
        >>> causal.integrated().round(3)
        array([[0.   , 0.154],
               [0.001, 0.   ]])
        >>> round(causal.band("rate", "gdp", low_period=6, high_period=32), 3)
        0.261
        >>> own = np.column_stack([np.ones(499), y[:-1, 0]])
        >>> both = np.column_stack([own, y[:-1, 1]])
        >>> fit = lambda x: y[1:, 0] - x @ np.linalg.lstsq(x, y[1:, 0], rcond=None)[0]
        >>> round(float(np.log(fit(own).var() / fit(both).var())), 3)
        0.153
    """

    names: tuple[str, ...]
    """Variable labels, indexing both matrix axes of ``measure``."""
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` uniform grid from 0 to :math:`\\pi`, endpoints included. Kept out of the repr."""
    measure: npt.NDArray[np.float64] = field(repr=False)
    """``(n, k, k)`` non-negative measure, ``[frequency, effect, cause]``, zero diagonal.

    Kept out of the repr.
    """

    @property
    def k_endog(self) -> int:
        """Number of variables, :math:`k`, the size of each matrix axis."""
        return len(self.names)

    def _index(self, name: str) -> int:
        """The axis index of a variable label, after checking it exists.

        Args:
            name: One of ``names``.

        Returns:
            Its position.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
            >>> causal = SpectralCausality(res, n_frequencies=16).compute()
            >>> causal._index("rate")
            1
            >>> causal._index("cpi")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown variable 'cpi'; the system has ...
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown variable {name!r}; the system has {self.names}."
            ) from None

    def pair(self, cause: str, effect: str) -> npt.NDArray[np.float64]:
        """One directed pair's measure across the grid.

        Args:
            cause: The variable whose history is being credited.
            effect: The variable whose spectrum is being decomposed.

        Returns:
            The ``(n,)`` non-negative measure.

        Raises:
            SpecificationError: If either label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((500, 2))
            >>> for t in range(1, 500):
            ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
            >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
            >>> causal = SpectralCausality(res, n_frequencies=64).compute()
            >>> curve = causal.pair("rate", "gdp")
            >>> curve.shape, bool(np.all(curve >= 0.0)), bool(curve[0] > curve[-1])
            ((64,), True, True)
            >>> bool(np.array_equal(curve, causal.measure[:, 0, 1]))
            True
        """
        return self.measure[:, self._index(effect), self._index(cause)]

    def integrated(self) -> npt.NDArray[np.float64]:
        r"""The measures integrated over frequency: the time-domain reading.

        .. math::

           F_{j \to i} = \frac{1}{\pi} \int_0^{\pi} f_{j \to i}(\omega)\, d\omega,

        which equals the average over the full circle by symmetry, and
        recovers Geweke's time-domain measure
        :math:`\ln(\sigma^2_{\text{restricted}} / \sigma^2_{\text{full}})`
        up to the approximation the literature documents. Trapezoidal
        on the grid.

        Returns:
            A ``(k, k)`` array, ``[effect, cause]``, zero diagonal.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((500, 2))
            >>> for t in range(1, 500):
            ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
            >>> causal = SpectralCausality(VAR(y, order=1).fit(), n_frequencies=64).compute()
            >>> totals = causal.integrated()
            >>> totals.shape, bool(totals[0, 1] > 10 * totals[1, 0])
            ((2, 2), True)
        """
        return np.asarray(
            np.trapezoid(self.measure, self.frequencies, axis=0) / np.pi,
            dtype=np.float64,
        )

    def band(self, cause: str, effect: str, *, low_period: float, high_period: float) -> float:
        r"""One pair's measure averaged over a period band.

        The mean of :math:`f_{j \to i}` over the grid frequencies inside
        :math:`[2\pi / \texttt{high\_period},\ 2\pi / \texttt{low\_period}]`,
        the business-cycle slice of the curve when the band is the
        Burns-Mitchell one.

        Args:
            cause: The variable whose history is being credited.
            effect: The variable whose spectrum is being decomposed.
            low_period: Shortest period in the band, in observations per
                cycle (6 for the quarterly business-cycle convention).
            high_period: Longest period in the band (32 quarterly).

        Returns:
            The average measure over frequencies ``[2 pi / high_period,
            2 pi / low_period]``.

        Raises:
            SpecificationError: If the band is malformed or misses the
                grid.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((500, 2))
            >>> for t in range(1, 500):
            ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
            >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
            >>> causal = SpectralCausality(res, n_frequencies=64).compute()
            >>> cycle = causal.band("rate", "gdp", low_period=6, high_period=32)
            >>> short = causal.band("rate", "gdp", low_period=2, high_period=4)
            >>> bool(cycle > short)
            True
            >>> causal.band("rate", "gdp", low_period=32, high_period=6)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: periods must satisfy 0 < low_period < ...
        """
        if not 0.0 < low_period < high_period:
            raise SpecificationError(
                f"periods must satisfy 0 < low_period < high_period; got "
                f"{low_period}, {high_period}."
            )
        lower = 2.0 * np.pi / high_period
        upper = 2.0 * np.pi / low_period
        inside = (self.frequencies >= lower) & (self.frequencies <= upper)
        if not np.any(inside):
            raise SpecificationError(
                "the period band contains no grid frequencies; widen the band or refine the grid."
            )
        return float(self.pair(cause, effect)[inside].mean())

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: the ten largest integrated pairs.

        Every ordered pair's integrated measure, sorted descending and
        cut at ten, under a header with the variable, grid, and pair
        counts; the notes say what the entries are, warn about common
        driving, and restate that the measures describe the fitted
        model.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> causal = SpectralCausality(VAR(y, order=1).fit(), n_frequencies=16).compute()
            >>> table = causal._summary_table()
            >>> table.columns, len(table.rows), table.metadata[2]
            (('pair (top 10 integrated)', 'integrated measure'), 6, ('Pairs', '6'))
        """
        totals = self.integrated()
        pairs = [
            (self.names[j], self.names[i], float(totals[i, j]))
            for i in range(self.k_endog)
            for j in range(self.k_endog)
            if i != j
        ]
        pairs.sort(key=lambda item: item[2], reverse=True)
        rows = tuple(
            (f"{cause} -> {effect}", f"{value:.4f}") for cause, effect, value in pairs[:10]
        )
        notes = [
            "Entries are Geweke's unconditional pairwise measures "
            "integrated over frequency -- the time-domain Granger reading; "
            "pair() has each full curve and band() the business-cycle "
            "slice.",
            "Pairwise unconditional measures can reflect common driving by "
            "a third variable; the conditional measure "
            "(ConditionalSpectralCausality) is the control for that.",
            "Statements about the fitted model, read as association across "
            "frequencies -- not causal claims about the world.",
        ]
        return SummaryTable(
            title="Spectral Granger Causality",
            metadata=(
                ("Variables", f"{self.k_endog}"),
                ("Grid", f"{len(self.frequencies)} frequencies on [0, pi]"),
                ("Pairs", f"{self.k_endog * (self.k_endog - 1)}"),
            ),
            columns=("pair (top 10 integrated)", "integrated measure"),
            rows=rows,
            notes=tuple(notes),
        )


class SpectralCausality:
    r"""Unconditional pairwise Geweke causality of a fitted closed system.

    Not an estimator: constructs with a fitted result and decomposes it.
    The system's transfer function :math:`\Psi(\omega) = [I - \sum_l A_l
    e^{-i\omega l}]^{-1}` and innovation covariance :math:`\Sigma` give
    its spectral density :math:`S(\omega) = \Psi \Sigma \Psi^\ast / 2\pi`.
    For a bivariate system the innovation representation is the model's
    own and the measure follows directly; for a larger one each pair's
    marginal process

    .. math::

       S_{\{i,j\}}(\omega) = \Psi_{\{i,j\}\cdot}(\omega)\, \Sigma\,
       \Psi_{\{i,j\}\cdot}(\omega)^\ast / 2\pi

    is generally VARMA and matches no lag stack, so its own innovation
    representation :math:`(H, \Sigma_{ij})` is recovered exactly from the
    marginal density by Wilson's spectral factorization -- an iterative
    solve, converging quadratically for the smooth densities stable
    models produce -- and the bivariate measure is computed from that.
    Nothing is re-estimated.

    Args:
        result: A fitted closed reduced-form result with at least two
            variables.
        n_frequencies: Grid points on ``[0, pi]``, endpoints included.

    Attributes:
        _result: The fitted result being decomposed.
        _n_frequencies: The grid size :math:`n`.

    Raises:
        SpecificationError: If the result is not a closed system, has
            fewer than two variables, or the grid is too small.

    Note:
        The grid is the caller's resolution choice, and two things depend
        on it: :meth:`SpectralCausalityResult.band` averages the grid
        points inside a period band and refuses a band that catches
        none, so a narrow band needs a fine grid; and the spectral
        factorization for :math:`k > 2` runs on the full circle of
        :math:`2(n - 1)` points, so its cost and its accuracy both grow
        with :math:`n`. The default of 256 resolves the quarterly
        business-cycle band into about forty points.

    See Also:
        * :class:`SpectralCausalityResult` -- the record
          :meth:`compute` returns.
        * :class:`ConditionalSpectralCausality` -- the conditional
          measure, which needs a second fitted model.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- a closed system whose ``granger_causality`` is the
          time-domain test of the same relation.

    References:
        Geweke, J. (1982). Measurement of linear dependence and feedback
        between multiple time series. *Journal of the American Statistical
        Association*, 77(378), 304-313.

        Dhamala, M., Rangarajan, G., & Ding, M. (2008). Estimating Granger
        causality from Fourier and wavelet transforms of time series data.
        *Physical Review Letters*, 100(1), 018701.

        Wilson, G. T. (1972). The factorization of matricial spectral
        densities. *SIAM Journal on Applied Mathematics*, 23(4), 420-426.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((500, 2))
        >>> for t in range(1, 500):
        ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
        ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
        >>> spectral = SpectralCausality(VAR(y, order=1).fit(), n_frequencies=64)
        >>> causal = spectral.compute()
        >>> forward = causal.integrated()[0, 1]
        >>> backward = causal.integrated()[1, 0]
        >>> bool(forward > 10 * backward)
        True

        A third, independent variable enters the system and the pairwise
        measures involving it stay near zero:

        >>> wider = np.column_stack([y, rng.standard_normal(500)])
        >>> three = SpectralCausality(VAR(wider, order=1).fit(), n_frequencies=32).compute()
        >>> three.measure.shape, bool(three.integrated()[:, 2].max() < 0.01)
        ((32, 3, 3), True)
    """

    __slots__ = ("_n_frequencies", "_result")

    def __init__(self, result: ClosedSystemResult, *, n_frequencies: int = 256) -> None:
        """Validate the source surface and the grid.

        The result is checked against the
        :class:`~cultivars._core.ClosedSystemResult` protocol at runtime,
        so a model that has not been fitted, or a univariate result, is
        refused by name.

        Args:
            result: Checked to be a closed system with at least two
                variables.
            n_frequencies: Checked to be at least two.

        Raises:
            SpecificationError: If the result is not a closed system, has
                fewer than two variables, or the grid is below two points.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((100, 2))
            >>> SpectralCausality(VAR(y, order=1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: SpectralCausality reads a fitted closed ...
            >>> SpectralCausality(VAR(y, order=1).fit(), n_frequencies=1)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n_frequencies must be at least 2; got 1.
        """
        if not isinstance(result, ClosedSystemResult):
            raise SpecificationError(
                "SpectralCausality reads a fitted closed reduced-form result "
                "exposing coefficients, sigma_u, and names; got "
                f"{type(result).__name__}."
            )
        if result.k_endog < 2:
            raise SpecificationError(
                f"spectral causality needs at least two variables; got {result.k_endog}."
            )
        if n_frequencies < 2:
            raise SpecificationError(f"n_frequencies must be at least 2; got {n_frequencies}.")
        self._result = result
        self._n_frequencies = int(n_frequencies)

    def compute(self) -> SpectralCausalityResult:
        """Decompose every ordered pair.

        The transfer function is evaluated on the full circle of
        ``2 (n - 1)`` points so that the factorization has the whole
        density; the measure is read back on the ``[0, pi]`` half. For
        ``k = 2`` the model's own transfer function and covariance feed
        ``_pairwise_measure`` directly, once in each direction by
        swapping the pair's axes; for ``k > 2`` each pair's marginal
        density is cut from the system's spectral matrix and factored
        first. Deterministic.

        Returns:
            The :class:`SpectralCausalityResult`.

        Raises:
            NumericalError: If a pair's spectral factorization fails, which
                for a stable fitted model means its density is degenerate.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> causal = SpectralCausality(VAR(y, order=1).fit(), n_frequencies=16).compute()
            >>> causal.frequencies.shape, bool(np.all(causal.measure >= 0.0))
            ((16,), True)
            >>> bool(np.all(np.diagonal(causal.measure, axis1=1, axis2=2) == 0.0))
            True
        """
        result = self._result
        k = result.k_endog
        n = self._n_frequencies
        half = np.pi * np.arange(n - 1) / (n - 1)
        circle = np.concatenate([half, np.pi + half])
        coefficients = np.asarray(result.coefficients)
        sigma_u = np.asarray(result.sigma_u)
        transfer = transfer_function(coefficients, circle)
        measure = np.zeros((n, k, k))
        grid = frequency_grid(n)
        if k == 2:
            forward = _pairwise_measure(transfer, sigma_u)
            swap = transfer[:, ::-1][:, :, ::-1]
            backward = _pairwise_measure(swap, sigma_u[::-1][:, ::-1])
            measure[:, 0, 1] = forward[:n]
            measure[:, 1, 0] = backward[:n]
            return SpectralCausalityResult(
                names=tuple(result.names), frequencies=grid, measure=measure
            )
        density = spectral_matrix(transfer, sigma_u)
        for i in range(k):
            for j in range(i + 1, k):
                select = np.array([i, j])
                marginal = density[:, select][:, :, select]
                pair_transfer, pair_sigma = _spectral_factor(marginal)
                forward = _pairwise_measure(pair_transfer, pair_sigma)
                swap = pair_transfer[:, ::-1][:, :, ::-1]
                backward = _pairwise_measure(swap, pair_sigma[::-1][:, ::-1])
                measure[:, i, j] = forward[:n]
                measure[:, j, i] = backward[:n]
        return SpectralCausalityResult(names=tuple(result.names), frequencies=grid, measure=measure)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ConditionalSpectralCausalityResult(_SummaryMixin):
    r"""Geweke's conditional measure from one cause to every other variable.

    For a cause :math:`x`, an effect :math:`y`, and the remaining
    variables :math:`z`, the conditional measure at frequency
    :math:`\omega` is

    .. math::

       f_{x \to y \mid z}(\omega) = \ln \frac{\Sigma^{r}_{yy}}
       {\bigl|Q_{yy}(\omega)\bigr|^2 \, \tilde\Sigma_{yy}}
       \;\ge\; 0,

    where :math:`\Sigma^{r}_{yy}` is :math:`y`'s innovation variance in
    the restricted system fitted without :math:`x`, and the denominator
    is the part of that innovation power the full system attributes to
    innovations other than :math:`x`'s once the full model's shocks are
    mapped through the restricted lag polynomial :math:`Q(\omega)`. What
    remains is the power in :math:`y`'s restricted innovation that
    traces back to :math:`x`'s shocks: :math:`x`'s contribution to
    :math:`y` given every other variable's history, which the pairwise
    unconditional measure cannot separate from common driving.

    Attributes:
        cause: The variable whose history is being credited.
        effect_names: The remaining variables, in the restricted model's
            order.
        frequencies: The ``(n,)`` grid on ``[0, pi]``.
        measure: The ``(n, k - 1)`` measure per effect.
        consistency: The largest relative gap between the transformed full
            model's implied innovation spectrum and the restricted model's
            innovation variance -- exactly zero in population, and a
            diagnostic for how compatible the two separately fitted models
            are. Large values mean the restricted model is badly specified
            relative to the full one, and the curves should be read with
            suspicion.

    Note:
        The measure is one cause against every effect, not a matrix: a
        different cause needs a different restricted model, so it is a
        different :class:`ConditionalSpectralCausality`. The
        ``consistency`` number has no threshold the literature supplies;
        it is the relative size of the mismatch between two models that
        would agree exactly if both were true, and values around a tenth
        are ordinary for a finite-order restricted VAR standing in for a
        VARMA process. A jump when the restricted lag order is reduced
        is the signal that the restricted model is too short.

    See Also:
        * :class:`ConditionalSpectralCausality` -- the producer.
        * :class:`SpectralCausalityResult` -- the unconditional pairwise
          measures, which need no second model.

    References:
        Geweke, J. (1984). Measures of conditional linear dependence and
        feedback between time series. *Journal of the American Statistical
        Association*, 79(388), 907-915.

        Ding, M., Chen, Y., & Bressler, S. L. (2006). Granger causality:
        Basic theory and application to neuroscience. In *Handbook of Time
        Series Analysis* (pp. 437-460). Wiley.

    Example:
        A third variable drives the first two; unconditionally the second
        looks causal for the first, and conditioning on the driver
        removes it:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> z = np.zeros((800, 3))
        >>> for t in range(1, 800):
        ...     z[t, 2] = 0.6 * z[t - 1, 2] + rng.standard_normal()
        ...     z[t, 0] = 0.3 * z[t - 1, 0] + 0.8 * z[t - 1, 2] + rng.standard_normal()
        ...     z[t, 1] = 0.3 * z[t - 1, 1] + 0.8 * z[t - 1, 2] + rng.standard_normal()
        >>> full = VAR(z, order=1).fit()
        >>> pairwise = SpectralCausality(full, n_frequencies=64).compute().integrated()
        >>> without = VAR(z[:, [0, 2]], order=3, names=("y1", "y3")).fit()
        >>> given = ConditionalSpectralCausality(full, without, n_frequencies=64).compute()
        >>> given.cause, given.effect_names, given.measure.shape
        ('y2', ('y1', 'y3'), (64, 2))
        >>> bool(pairwise[0, 1] > 0.01), bool(given.integrated()[0] < 0.001)
        (True, True)
        >>> bool(given.consistency < 0.2)
        True
    """

    cause: str
    """The variable whose history is being credited, the one the restricted model lacks."""
    effect_names: tuple[str, ...]
    """The remaining variables, in the restricted model's order; the columns of ``measure``."""
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` uniform grid from 0 to :math:`\\pi`, endpoints included. Kept out of the repr."""
    measure: npt.NDArray[np.float64] = field(repr=False)
    """``(n, k - 1)`` non-negative conditional measure, one column per effect.

    Kept out of the repr.
    """
    consistency: float
    """Largest relative gap between the two models' implied innovation power; zero in population."""

    def _index(self, name: str) -> int:
        """The column index of an effect label, after checking it exists.

        Args:
            name: One of ``effect_names``.

        Returns:
            Its position.

        Raises:
            SpecificationError: If the label is unknown, including when it
                is the cause.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> full = VAR(y, order=1).fit()
            >>> without = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
            >>> given = ConditionalSpectralCausality(full, without, n_frequencies=16).compute()
            >>> given._index("y3")
            1
            >>> given._index("y2")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown effect 'y2'; the conditional system ...
        """
        try:
            return self.effect_names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown effect {name!r}; the conditional system has {self.effect_names}."
            ) from None

    def pair(self, effect: str) -> npt.NDArray[np.float64]:
        """The conditional measure ``cause -> effect`` across the grid.

        Args:
            effect: One of ``effect_names``.

        Returns:
            The ``(n,)`` non-negative measure.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> full = VAR(y, order=1).fit()
            >>> without = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
            >>> given = ConditionalSpectralCausality(full, without, n_frequencies=16).compute()
            >>> curve = given.pair("y1")
            >>> curve.shape, bool(np.array_equal(curve, given.measure[:, 0]))
            ((16,), True)
        """
        return self.measure[:, self._index(effect)]

    def integrated(self) -> npt.NDArray[np.float64]:
        r"""The measures integrated over frequency, one value per effect.

        :math:`\frac{1}{\pi} \int_0^{\pi} f_{x \to y \mid z}(\omega)\,
        d\omega` by the trapezoidal rule on the grid, which recovers
        Geweke's time-domain conditional measure up to the two models'
        compatibility.

        Returns:
            A ``(k - 1,)`` array in ``effect_names`` order.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((600, 3))
            >>> for t in range(1, 600):
            ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 2] = 0.5 * y[t - 1, 2] + rng.standard_normal()
            >>> full = VAR(y, order=1).fit()
            >>> without = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
            >>> ConditionalSpectralCausality(full, without).compute().integrated().round(2)
            array([0.16, 0.  ])
        """
        return np.asarray(
            np.trapezoid(self.measure, self.frequencies, axis=0) / np.pi,
            dtype=np.float64,
        )

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: one row per effect, largest first.

        The integrated measure of the cause on each effect, sorted
        descending, under a header with the cause, the effect count, and
        the grid; the notes say what the measure is, report the
        consistency diagnostic, and restate that the curves describe the
        fitted models.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> full = VAR(y, order=1).fit()
            >>> without = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
            >>> table = ConditionalSpectralCausality(full, without).compute()._summary_table()
            >>> table.columns, len(table.rows), table.metadata[0]
            (('pair', 'integrated measure'), 2, ('Cause', 'y2'))
        """
        totals = self.integrated()
        order = np.argsort(totals)[::-1]
        rows = tuple((f"{self.cause} -> {self.effect_names[i]}", f"{totals[i]:.4f}") for i in order)
        notes = [
            "Geweke's conditional measure: the cause's contribution to each "
            "effect's spectrum given every other variable's history -- the "
            "control that pairwise unconditional measures lack.",
            "Computed from two separately fitted models (the full system "
            "and the caller's restricted one); the consistency diagnostic "
            f"is {self.consistency:.2e}, exactly zero in population.",
            "Statements about the fitted models, read as association "
            "across frequencies -- not causal claims about the world.",
        ]
        return SummaryTable(
            title=f"Conditional Spectral Causality of {self.cause}",
            metadata=(
                ("Cause", self.cause),
                ("Effects", f"{len(self.effect_names)}"),
                ("Grid", f"{len(self.frequencies)} frequencies on [0, pi]"),
            ),
            columns=("pair", "integrated measure"),
            rows=rows,
            notes=tuple(notes),
        )


class ConditionalSpectralCausality:
    r"""Geweke's (1984) conditional causality, from two fitted models.

    The conditional measure mathematically requires the restricted system
    -- the model with the cause excluded -- and there is no view-only
    shortcut, so that model is an explicit argument the caller fits. With
    :math:`\Psi(\omega)` the full system's transfer function,
    :math:`\Sigma` its innovation covariance, and :math:`Q(\omega) = I -
    \sum_l B_l e^{-i\omega l}` the restricted system's lag polynomial
    embedded in the full dimension with the cause's row and column left
    as the identity, the full model's innovations are mapped through the
    restricted polynomial,

    .. math::

       G(\omega) = Q(\omega)\, \Psi(\omega)\, P,

    where :math:`P` is the Cholesky factor of :math:`\Sigma` with the
    cause's shock ordered *last*, the conservative convention that
    credits every shared innovation to the other variables first. Row
    :math:`y` of :math:`G` then splits the restricted innovation power of
    effect :math:`y` into the part from every shock but the cause's and
    the part from the cause's, and the measure is the log ratio of the
    whole to the former. In population the whole equals
    :math:`\Sigma^{r}_{yy}` at every frequency; the largest relative
    departure is reported as ``consistency``.

    Args:
        result: The fitted full system.
        restricted: A fitted closed system on the same variables minus
            exactly one -- the cause -- in the same relative order. The
            true restricted process is generally VARMA even when the full
            system is a finite VAR, so give this model a generous lag
            order and read the result's ``consistency`` diagnostic; a
            too-short restricted model is the usual reason the curves and
            the time-domain measure disagree.
        n_frequencies: Grid points on ``[0, pi]``, endpoints included.

    Attributes:
        _result: The fitted full system.
        _restricted: The fitted restricted system.
        _cause_index: The cause's column in the full system.
        _n_frequencies: The grid size :math:`n`.

    Raises:
        SpecificationError: If the two systems' variables do not differ by
            exactly one, order included, or the grid is too small.

    Note:
        Which variable is the cause is inferred from the names: the one
        the full system has and the restricted lacks. The two models are
        fitted separately by the caller and nothing here enforces that
        they were fitted to the same sample; a restricted model on a
        different window is a different process, and the consistency
        diagnostic will say so loudly.

    See Also:
        * :class:`ConditionalSpectralCausalityResult` -- the record
          :meth:`compute` returns.
        * :class:`SpectralCausality` -- the unconditional pairwise
          measures from one model.

    References:
        Geweke, J. (1984). Measures of conditional linear dependence and
        feedback between time series. *Journal of the American Statistical
        Association*, 79(388), 907-915.

        Chen, Y., Bressler, S. L., & Ding, M. (2006). Frequency
        decomposition of conditional Granger causality and application to
        multivariate neural field potential data. *Journal of Neuroscience
        Methods*, 150(2), 228-237.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((600, 3))
        >>> for t in range(1, 600):
        ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
        ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
        ...     y[t, 2] = 0.5 * y[t - 1, 2] + rng.standard_normal()
        >>> full = VAR(y, order=1).fit()
        >>> restricted = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
        >>> conditional = ConditionalSpectralCausality(full, restricted).compute()
        >>> conditional.cause
        'y2'
        >>> bool(conditional.integrated()[0] > 0.05)
        True

        The restricted system must be the full one minus exactly one
        variable, in order:

        >>> swapped = VAR(y[:, [2, 0]], order=1, names=("y3", "y1")).fit()
        >>> ConditionalSpectralCausality(full, swapped)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the restricted system must contain the full ...
    """

    __slots__ = ("_cause_index", "_n_frequencies", "_restricted", "_result")

    def __init__(
        self,
        result: ClosedSystemResult,
        restricted: ClosedSystemResult,
        *,
        n_frequencies: int = 256,
    ) -> None:
        """Validate both surfaces and their variable alignment.

        Both results are checked against the
        :class:`~cultivars._core.ClosedSystemResult` protocol, the grid
        against its minimum, and the restricted system's names against
        the full system's with exactly one removed and the rest in the
        same relative order; that removed name is the cause.

        Args:
            result: Checked to be a fitted closed system.
            restricted: Checked to be a fitted closed system on the full
                system's variables minus one.
            n_frequencies: Checked to be at least two.

        Raises:
            SpecificationError: If either result is not a closed system,
                the grid is below two points, or the variables do not
                differ by exactly one in order.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> full = VAR(y, order=1).fit()
            >>> ConditionalSpectralCausality(full, full)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the restricted system must contain the full ...
            >>> ConditionalSpectralCausality(VAR(y, order=1), full)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: result must be a fitted closed reduced-form ...
        """
        for label, candidate in (("result", result), ("restricted", restricted)):
            if not isinstance(candidate, ClosedSystemResult):
                raise SpecificationError(
                    f"{label} must be a fitted closed reduced-form result; "
                    f"got {type(candidate).__name__}."
                )
        if n_frequencies < 2:
            raise SpecificationError(f"n_frequencies must be at least 2; got {n_frequencies}.")
        missing = [name for name in result.names if name not in restricted.names]
        if len(missing) != 1 or tuple(restricted.names) != tuple(
            name for name in result.names if name != missing[0]
        ):
            raise SpecificationError(
                "the restricted system must contain the full system's "
                "variables minus exactly one -- the cause -- in the same "
                f"relative order; full has {result.names}, restricted has "
                f"{restricted.names}."
            )
        self._result = result
        self._restricted = restricted
        self._cause_index = result.names.index(missing[0])
        self._n_frequencies = int(n_frequencies)

    def compute(self) -> ConditionalSpectralCausalityResult:
        r"""Decompose every effect's restricted innovation power.

        Evaluates the full transfer function on the grid, builds the
        restricted lag polynomial :math:`Q(\omega)` and embeds it in the
        full dimension, forms :math:`G = Q \Psi P` with the cause's
        shock ordered last in the Cholesky factor, and for each effect
        splits :math:`\sum_s |G_{ys}|^2` into the cause's column and the
        rest. The consistency diagnostic compares the total against the
        restricted innovation variance. Deterministic.

        Returns:
            The :class:`ConditionalSpectralCausalityResult`.

        Raises:
            NumericalError: If the full system's innovation covariance is
                not positive definite, so the Cholesky factor does not
                exist.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> full = VAR(y, order=1).fit()
            >>> without = VAR(y[:, [0, 2]], order=1, names=("y1", "y3")).fit()
            >>> given = ConditionalSpectralCausality(full, without, n_frequencies=16).compute()
            >>> given.frequencies.shape, given.measure.shape, bool(np.all(given.measure >= 0.0))
            ((16,), (16, 2), True)
        """
        result, restricted = self._result, self._restricted
        k, cause = result.k_endog, self._cause_index
        others = [index for index in range(k) if index != cause]
        grid = frequency_grid(self._n_frequencies)
        n = grid.shape[0]
        full_transfer = transfer_function(np.asarray(result.coefficients), grid)
        sigma_full = np.asarray(result.sigma_u)
        stack = np.asarray(restricted.coefficients)
        polynomial = np.tile(np.eye(k - 1, dtype=np.complex128), (n, 1, 1))
        for lag in range(1, stack.shape[0] + 1):
            phase = np.exp(-1j * grid * lag)
            polynomial -= phase[:, None, None] * stack[lag - 1]
        embedded = np.tile(np.eye(k, dtype=np.complex128), (n, 1, 1))
        rows = np.asarray(others)
        embedded[:, rows[:, None], rows[None, :]] = polynomial
        mapped = embedded @ full_transfer
        permutation = [*others, cause]
        cholesky = np.linalg.cholesky(sigma_full[np.ix_(permutation, permutation)])
        loading = np.zeros((k, k))
        for position, original in enumerate(permutation):
            loading[original] = cholesky[position]
        shocks = np.asarray(mapped @ loading, dtype=np.complex128)
        total = np.sum(np.abs(shocks) ** 2, axis=2)
        causal = np.abs(shocks[:, :, k - 1]) ** 2
        sigma_restricted = np.asarray(restricted.sigma_u)
        measure = np.zeros((n, k - 1))
        drift = 0.0
        for position, original in enumerate(others):
            own_total = total[:, original]
            own_causal = causal[:, original]
            measure[:, position] = np.log(own_total / np.maximum(own_total - own_causal, 1e-300))
            flat = float(sigma_restricted[position, position])
            drift = max(drift, float(np.abs(own_total - flat).max()) / max(flat, 1e-300))
        return ConditionalSpectralCausalityResult(
            cause=result.names[cause],
            effect_names=tuple(restricted.names),
            frequencies=grid,
            measure=measure,
            consistency=drift,
        )
