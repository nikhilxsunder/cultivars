# filepath: /src/cultivars/spectral/density.py
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

"""The spectral density of a fitted system: the same model, read by frequency.

A fitted closed system already contains its complete frequency-domain story:
the spectral density matrix ``f(omega) = Psi(omega) Sigma_u Psi(omega)* /
(2 pi)`` is a closed form of the coefficients and the innovation covariance,
nothing more. This module reads it out -- per-series spectra saying where in
the cycle each variable's variance lives, coherence saying how tightly two
variables move at each frequency, partial coherence conditioning that
comovement on everything else, and gain and phase giving the lead-lag
reading. In the package's grammar this is a *view*, exactly like the
spillover table: construct with any fitted closed reduced-form result -- an
OLS VAR, a shrunk BVAR at its posterior mean, a sparse hundred-variable
system -- and nothing is re-estimated.

One honesty rule frames every number: a spectrum computed from a fitted
model is a statement about *the fitted model*, inheriting every
specification error the time-domain fit carries. No nonparametric
periodogram estimation is offered alongside, precisely so the package never
blurs which of the two objects a number came from.

References:
    Geweke, J. (1982). Measurement of linear dependence and feedback
        between multiple time series. *Journal of the American Statistical
        Association*, 77(378), 304-313.
    Hamilton, J. D. (1994). *Time Series Analysis*, chapter 6. Princeton
        University Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    ClosedSystemResult,
    SummaryTable,
    frequency_grid,
    spectral_matrix,
    transfer_function,
)
from .._internals import _SummaryMixin
from ..exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralDensityResult(_SummaryMixin):
    """A fitted system's spectral density matrix, and the readings off it.

    All frequencies are radians per observation on ``[0, pi]``; the process
    is real, so the negative half-circle is the conjugate mirror. The
    density carries the ``1 / (2 pi)`` normalization: integrating it over
    ``[-pi, pi]`` returns the model's unconditional autocovariance at lag
    zero.

    Attributes:
        names: Variable labels, indexing the matrix axes.
        frequencies: The ``(n,)`` grid on ``[0, pi]``.
        density: The ``(n, k, k)`` complex Hermitian density matrix.
    """

    names: tuple[str, ...]
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    density: npt.NDArray[np.complex128] = field(repr=False)

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

    def periods(self) -> npt.NDArray[np.float64]:
        """Each grid frequency as a period, in observations per cycle.

        ``2 pi / omega``, with frequency zero mapping to infinity: for
        quarterly data the business-cycle band of 6 to 32 quarters is
        frequencies ``2 pi / 32`` to ``2 pi / 6``.
        """
        with np.errstate(divide="ignore"):
            return np.asarray(
                np.where(
                    self.frequencies > 0.0,
                    2.0 * np.pi / np.maximum(self.frequencies, 1e-300),
                    np.inf,
                ),
                dtype=np.float64,
            )

    def spectrum(self, name: str) -> npt.NDArray[np.float64]:
        """One variable's power spectrum.

        The diagonal of the density matrix: where in frequency this
        variable's variance lives. Twice its integral over ``[0, pi]`` is
        the model's unconditional variance of the variable.

        Args:
            name: An endogenous variable.

        Returns:
            A real ``(n,)`` array.
        """
        index = self._index(name)
        return np.real(self.density[:, index, index])

    def coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        """Squared coherence between two variables, in ``[0, 1]``.

        ``|f_ij|**2 / (f_ii f_jj)``: the frequency-domain analogue of a
        squared correlation -- how much of the two variables' power at each
        frequency is shared. Symmetric, and silent about direction; the
        directional reading belongs to
        :class:`~cultivars.spectral.SpectralCausality`.

        Args:
            first: An endogenous variable.
            second: Another endogenous variable.

        Returns:
            A real ``(n,)`` array in ``[0, 1]``.
        """
        i, j = self._index(first), self._index(second)
        cross = np.abs(self.density[:, i, j]) ** 2
        power = np.real(self.density[:, i, i]) * np.real(self.density[:, j, j])
        return np.asarray(cross / np.maximum(power, 1e-300), dtype=np.float64)

    def partial_coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        """Squared partial coherence, conditioning on all other variables.

        Read off the inverse density matrix ``g = f**(-1)`` as
        ``|g_ij|**2 / (g_ii g_jj)``: the shared power between the two
        variables at each frequency once every other variable's
        contribution is removed -- the frequency-domain partial
        correlation, squared.

        Args:
            first: An endogenous variable.
            second: Another endogenous variable.

        Returns:
            A real ``(n,)`` array in ``[0, 1]``.
        """
        i, j = self._index(first), self._index(second)
        inverse = np.linalg.inv(self.density)
        cross = np.abs(inverse[:, i, j]) ** 2
        power = np.real(inverse[:, i, i]) * np.real(inverse[:, j, j])
        return np.asarray(cross / np.maximum(power, 1e-300), dtype=np.float64)

    def gain(self, effect: str, cause: str) -> npt.NDArray[np.float64]:
        """The gain of the regression of ``effect`` on ``cause`` by frequency.

        ``|f_ij| / f_jj`` with ``i`` the effect and ``j`` the cause: the
        amplitude multiplier the best linear frequency-wise predictor of
        ``effect`` from ``cause`` applies at each frequency.

        Args:
            effect: The variable being explained.
            cause: The variable explaining it.

        Returns:
            A real non-negative ``(n,)`` array.
        """
        i, j = self._index(effect), self._index(cause)
        return np.asarray(
            np.abs(self.density[:, i, j]) / np.maximum(np.real(self.density[:, j, j]), 1e-300),
            dtype=np.float64,
        )

    def phase(self, effect: str, cause: str) -> npt.NDArray[np.float64]:
        """The phase of the cross-spectrum ``f[effect, cause]``, in radians.

        The sign convention follows the transfer convention
        ``e**(-i omega l)``: a *positive* phase at frequency ``omega``
        means ``cause`` leads ``effect`` by ``phase / omega`` observations
        at that frequency. Half the literature states the mirror
        convention, so check signs against a known lead before trusting a
        plot.

        Args:
            effect: The variable being explained.
            cause: The variable explaining it.

        Returns:
            A real ``(n,)`` array in ``(-pi, pi]``.
        """
        i, j = self._index(effect), self._index(cause)
        return np.asarray(np.angle(self.density[:, i, j]), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = []
        interior = slice(1, None)
        for name in self.names:
            power = self.spectrum(name)
            peak = 1 + int(np.argmax(power[interior]))
            frequency = float(self.frequencies[peak])
            period = 2.0 * np.pi / frequency if frequency > 0.0 else np.inf
            share = float(
                np.trapezoid(power, self.frequencies)
                * 2.0
                / max(
                    float(
                        np.trapezoid(
                            np.real(np.trace(self.density, axis1=1, axis2=2)),
                            self.frequencies,
                        )
                        * 2.0
                    ),
                    1e-300,
                )
            )
            rows.append((name, f"{frequency:.4f}", f"{period:.1f}", f"{100.0 * share:.1f}%"))
        notes = [
            "The density is the fitted model's closed form, not a "
            "periodogram: it inherits every specification error the "
            "time-domain fit carries, and no nonparametric estimate is "
            "offered alongside.",
            "Frequencies are radians per observation on [0, pi]; twice the "
            "integral of a spectrum over the grid is that variable's "
            "model-implied unconditional variance.",
            "Peak frequency excludes the zero-frequency point, where a "
            "near-integrated system's density legitimately diverges.",
        ]
        return SummaryTable(
            title="Spectral Density",
            metadata=(
                ("Variables", f"{self.k_endog}"),
                ("Grid", f"{len(self.frequencies)} frequencies on [0, pi]"),
            ),
            columns=("variable", "peak frequency", "peak period", "power share"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class SpectralDensity:
    """The frequency-domain view of any fitted closed reduced form.

    Not an estimator: constructs with a fitted result -- anything exposing
    the closed-system surface -- and evaluates its exact spectral density.

    Args:
        result: A fitted closed reduced-form result.
        n_frequencies: Grid points on ``[0, pi]``, endpoints included.

    Raises:
        SpecificationError: If the result is not a closed system or the
            grid is too small.

    Example:
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((400, 2))
        >>> for t in range(1, 400):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
        >>> spec = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=128).compute()
        >>> spec.density.shape
        (128, 2, 2)
        >>> bool(np.all(spec.coherence("y1", "y2") <= 1.0 + 1e-12))
        True
    """

    __slots__ = ("_n_frequencies", "_result")

    def __init__(self, result: ClosedSystemResult, *, n_frequencies: int = 256) -> None:
        """Validate the source surface and the grid."""
        if not isinstance(result, ClosedSystemResult):
            raise SpecificationError(
                "SpectralDensity reads a fitted closed reduced-form result "
                "exposing coefficients, sigma_u, and names; got "
                f"{type(result).__name__}."
            )
        if n_frequencies < 2:
            raise SpecificationError(f"n_frequencies must be at least 2; got {n_frequencies}.")
        self._result = result
        self._n_frequencies = int(n_frequencies)

    def compute(self) -> SpectralDensityResult:
        """Evaluate the exact density on the grid.

        Returns:
            The :class:`SpectralDensityResult`.
        """
        result = self._result
        grid = frequency_grid(self._n_frequencies)
        transfer = transfer_function(np.asarray(result.coefficients), grid)
        return SpectralDensityResult(
            names=tuple(result.names),
            frequencies=grid,
            density=spectral_matrix(transfer, np.asarray(result.sigma_u)),
        )
