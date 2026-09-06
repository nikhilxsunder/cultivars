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

"""Geweke's causality by frequency: *when* one variable's history matters.

A time-domain Granger statement answers whether x's history helps predict
y; Geweke's (1982) spectral decomposition answers at which frequencies. The
measure ``f(omega)`` decomposes y's spectrum at each frequency into an
intrinsic part and a part attributable to x's innovations, and its integral
over the circle recovers the time-domain measure -- so a flat causality
curve and a business-cycle-peaked one can carry the same headline Granger
number while saying entirely different things about the economics.

Two objects, one honesty split. :class:`SpectralCausality` computes the
*unconditional* pairwise measure for every ordered pair, purely as a view
of one fitted system: for a bivariate model the transfer function and
covariance are the model's own; for a larger one, each pair's measure needs
the innovation representation of that pair's *marginal* process, which is
generally VARMA and matches no lag stack -- so it is recovered exactly from
the pair's marginal spectral density by Wilson's spectral factorization
(the route of Dhamala, Rangarajan & Ding 2008), still with nothing
re-estimated. :class:`ConditionalSpectralCausality` computes Geweke's
(1984) conditional measure, which mathematically requires a second,
restricted model -- there is no view-only shortcut -- so that model is an
explicit argument the caller fits, never something estimated silently
inside a "view".

The family caveat applies with force: these are statements about fitted
models, read as association across frequencies, not causal claims about the
world; and pairwise unconditional measures can reflect common driving by a
third variable, which is exactly what the conditional measure is for.

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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralCausalityResult(_SummaryMixin):
    """All ordered pairs' unconditional Geweke measures across frequency.

    Attributes:
        names: Variable labels, indexing the measure's matrix axes.
        frequencies: The ``(n,)`` grid on ``[0, pi]``.
        measure: The ``(n, k, k)`` measure; entry ``[w, i, j]`` is the
            causality from variable ``j`` to variable ``i`` at frequency
            ``w``. The diagonal is zero by convention.
    """

    names: tuple[str, ...]
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    measure: npt.NDArray[np.float64] = field(repr=False)

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return len(self.names)

    def _index(self, name: str) -> int:
        """Resolve a variable label.

        Raises:
            SpecificationError: If the label is unknown.
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
        """
        return self.measure[:, self._index(effect), self._index(cause)]

    def integrated(self) -> npt.NDArray[np.float64]:
        """The measures integrated over frequency: the time-domain reading.

        ``(1 / pi) * integral over [0, pi]``, which equals the average over
        the full circle by symmetry, and recovers Geweke's time-domain
        measure ``ln(sigma_restricted**2 / sigma_full**2)`` up to the
        approximation the literature documents.

        Returns:
            A ``(k, k)`` array, ``[effect, cause]``, zero diagonal.
        """
        return np.asarray(
            np.trapezoid(self.measure, self.frequencies, axis=0) / np.pi,
            dtype=np.float64,
        )

    def band(self, cause: str, effect: str, *, low_period: float, high_period: float) -> float:
        """One pair's measure averaged over a period band.

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
        """Build the structured summary."""
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
    """Unconditional pairwise Geweke causality of a fitted closed system.

    Not an estimator: constructs with a fitted result and decomposes it.
    For a bivariate system the innovation representation is the model's
    own; for a larger one each pair's marginal representation is recovered
    exactly from the pair's marginal spectral density by Wilson's spectral
    factorization -- an iterative solve, converging quadratically for the
    smooth densities stable models produce.

    Args:
        result: A fitted closed reduced-form result with at least two
            variables.
        n_frequencies: Grid points on ``[0, pi]``, endpoints included.

    Raises:
        SpecificationError: If the result is not a closed system, has
            fewer than two variables, or the grid is too small.

    Example:
        >>> from cultivars.multivariate.reduced_form import VAR
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
    """

    __slots__ = ("_n_frequencies", "_result")

    def __init__(self, result: ClosedSystemResult, *, n_frequencies: int = 256) -> None:
        """Validate the source surface and the grid."""
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

        Returns:
            The :class:`SpectralCausalityResult`.

        Raises:
            NumericalError: If a pair's spectral factorization fails, which
                for a stable fitted model means its density is degenerate.
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
    """Geweke's conditional measure from one cause to every other variable.

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
    """

    cause: str
    effect_names: tuple[str, ...]
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    measure: npt.NDArray[np.float64] = field(repr=False)
    consistency: float

    def _index(self, name: str) -> int:
        """Resolve an effect label.

        Raises:
            SpecificationError: If the label is unknown.
        """
        try:
            return self.effect_names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown effect {name!r}; the conditional system has {self.effect_names}."
            ) from None

    def pair(self, effect: str) -> npt.NDArray[np.float64]:
        """The conditional measure ``cause -> effect`` across the grid."""
        return self.measure[:, self._index(effect)]

    def integrated(self) -> npt.NDArray[np.float64]:
        """The measures integrated over frequency, one value per effect."""
        return np.asarray(
            np.trapezoid(self.measure, self.frequencies, axis=0) / np.pi,
            dtype=np.float64,
        )

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
    """Geweke's (1984) conditional causality, from two fitted models.

    The conditional measure mathematically requires the restricted system
    -- the model with the cause excluded -- and there is no view-only
    shortcut, so that model is an explicit argument the caller fits. The
    full model's innovations are mapped through the restricted model's lag
    polynomial; whatever part of the restricted system's innovation power
    traces back to the cause's shocks (attributed by ordering the cause's
    shock last, the conservative convention) is the conditional causality.

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

    Raises:
        SpecificationError: If the two systems' variables do not differ by
            exactly one, order included, or the grid is too small.

    Example:
        >>> from cultivars.multivariate.reduced_form import VAR
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
    """

    __slots__ = ("_cause_index", "_n_frequencies", "_restricted", "_result")

    def __init__(
        self,
        result: ClosedSystemResult,
        restricted: ClosedSystemResult,
        *,
        n_frequencies: int = 256,
    ) -> None:
        """Validate both surfaces and their variable alignment."""
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
        """Decompose every effect's restricted innovation power.

        Returns:
            The :class:`ConditionalSpectralCausalityResult`.
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
