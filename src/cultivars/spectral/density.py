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

__all__ = ["SpectralDensity", "SpectralDensityResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectralDensityResult(_SummaryMixin):
    r"""A fitted system's spectral density matrix, and the readings off it.

    The matrix is the closed form

    .. math::

       f(\omega) = \frac{1}{2\pi}\, \Psi(\omega)\, \Sigma_u\, \Psi(\omega)^\ast,
       \qquad
       \Psi(\omega) = \Bigl[I - \sum_{l=1}^{p} A_l e^{-i\omega l}\Bigr]^{-1},

    evaluated on a grid; every reading here is a function of its
    entries. All frequencies are radians per observation on ``[0, pi]``;
    the process is real, so the negative half-circle is the conjugate
    mirror. The density carries the :math:`1 / 2\pi` normalization:
    integrating it over :math:`[-\pi, \pi]` returns the model's
    unconditional autocovariance at lag zero, so twice the integral of a
    diagonal entry over :math:`[0, \pi]` is that variable's variance.

    Attributes:
        names: Variable labels, indexing the matrix axes.
        frequencies: The ``(n,)`` grid on ``[0, pi]``.
        density: The ``(n, k, k)`` complex Hermitian density matrix.

    Note:
        Every number is a statement about the fitted model, not about
        the data: the density inherits whatever the time-domain fit got
        wrong, and a spectral peak here is where the *model* puts
        variance. A near-integrated system has a density that diverges
        at frequency zero, legitimately, and the summary's peak search
        skips that point for that reason. The grid is uniform, so the
        long periods are coarsely resolved: with 256 points on
        :math:`[0, \pi]` the spacing is about :math:`\pi / 255`, which
        at the business-cycle end (periods 6 to 32) is twenty or so
        points and at periods above 100 only two or three.

    See Also:
        * :class:`SpectralDensity` -- the producer.
        * :class:`~cultivars.spectral.causality.SpectralCausalityResult`
          -- the directional decomposition of the same matrix.
        * :class:`~cultivars.spectral.periodogram.SpectrumEstimate` --
          the nonparametric estimate from the data, for the comparison
          this module deliberately does not make for you.

    References:
        Hamilton, J. D. (1994). *Time Series Analysis*, chapter 6.
        Princeton University Press.

        Priestley, M. B. (1981). *Spectral Analysis and Time Series*.
        Academic Press.

    Example:
        A rate that feeds into output one period ahead; the spectrum
        integrates to the model's variance, the two are coherent at low
        frequencies, and the phase says the rate leads:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((500, 2))
        >>> for t in range(1, 500):
        ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
        ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
        >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
        >>> density = SpectralDensity(res, n_frequencies=64).compute()
        >>> density.density.shape, density.k_endog
        ((64, 2, 2), 2)
        >>> variance = 2.0 * np.trapezoid(density.spectrum("gdp"), density.frequencies)
        >>> bool(abs(variance - y[:, 0].var()) < 0.05)
        True
        >>> density.coherence("gdp", "rate")[[1, -1]].round(2)
        array([0.35, 0.07])
        >>> bool(density.phase("gdp", "rate")[1] < 0.0)
        True
    """

    names: tuple[str, ...]
    """Variable labels, indexing both matrix axes of ``density``."""
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` uniform grid from 0 to :math:`\\pi`, endpoints included. Kept out of the repr."""
    density: npt.NDArray[np.complex128] = field(repr=False)
    """``(n, k, k)`` complex Hermitian density, :math:`\\Psi \\Sigma_u \\Psi^\\ast / 2\\pi`.

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
            >>> density = SpectralDensity(res, n_frequencies=16).compute()
            >>> density._index("rate")
            1
            >>> density._index("cpi")  # doctest: +ELLIPSIS
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

    def periods(self) -> npt.NDArray[np.float64]:
        r"""Each grid frequency as a period, in observations per cycle.

        :math:`2\pi / \omega`, with frequency zero mapping to infinity:
        for quarterly data the business-cycle band of 6 to 32 quarters is
        frequencies :math:`2\pi / 32` to :math:`2\pi / 6`.

        Returns:
            A ``(n,)`` array, ``inf`` first and ``2.0`` last.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> density = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=5).compute()
            >>> density.periods().round(2).tolist()
            [inf, 8.0, 4.0, 2.67, 2.0]
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
        r"""One variable's power spectrum.

        The diagonal of the density matrix, :math:`f_{ii}(\omega)`: where
        in frequency this variable's variance lives. Twice its integral
        over :math:`[0, \pi]` is the model's unconditional variance of the
        variable.

        Args:
            name: An endogenous variable.

        Returns:
            A real ``(n,)`` array.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((500, 2))
            >>> for t in range(1, 500):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> density = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=64).compute()
            >>> power = density.spectrum("y1")
            >>> power.shape, bool(power[0] > power[-1])
            ((64,), True)
        """
        index = self._index(name)
        return np.real(self.density[:, index, index])

    def coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        r"""Squared coherence between two variables, in ``[0, 1]``.

        .. math::

           C_{ij}(\omega) = \frac{|f_{ij}(\omega)|^2}{f_{ii}(\omega)\, f_{jj}(\omega)},

        the frequency-domain analogue of a squared correlation -- how
        much of the two variables' power at each frequency is shared.
        Symmetric, and silent about direction; the directional reading
        belongs to :class:`~cultivars.spectral.causality.SpectralCausality`.

        Args:
            first: An endogenous variable.
            second: Another endogenous variable.

        Returns:
            A real ``(n,)`` array in ``[0, 1]``.

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
            >>> density = SpectralDensity(res, n_frequencies=64).compute()
            >>> forward = density.coherence("gdp", "rate")
            >>> bool(np.allclose(forward, density.coherence("rate", "gdp")))
            True
            >>> bool(np.all((forward >= 0.0) & (forward <= 1.0)))
            True
        """
        i, j = self._index(first), self._index(second)
        cross = np.abs(self.density[:, i, j]) ** 2
        power = np.real(self.density[:, i, i]) * np.real(self.density[:, j, j])
        return np.asarray(cross / np.maximum(power, 1e-300), dtype=np.float64)

    def partial_coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        r"""Squared partial coherence, conditioning on all other variables.

        Read off the inverse density matrix :math:`g = f^{-1}` as

        .. math::

           \tilde C_{ij}(\omega) = \frac{|g_{ij}(\omega)|^2}{g_{ii}(\omega)\, g_{jj}(\omega)},

        the shared power between the two variables at each frequency once
        every other variable's contribution is removed -- the
        frequency-domain partial correlation, squared. For a bivariate
        system there is nothing to condition on and it equals
        :meth:`coherence`.

        Args:
            first: An endogenous variable.
            second: Another endogenous variable.

        Returns:
            A real ``(n,)`` array in ``[0, 1]``.

        Raises:
            SpecificationError: If either label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> z = np.zeros((800, 3))
            >>> for t in range(1, 800):
            ...     z[t, 2] = 0.6 * z[t - 1, 2] + rng.standard_normal()
            ...     z[t, 0] = 0.3 * z[t - 1, 0] + 0.8 * z[t - 1, 2] + rng.standard_normal()
            ...     z[t, 1] = 0.3 * z[t - 1, 1] + 0.8 * z[t - 1, 2] + rng.standard_normal()
            >>> density = SpectralDensity(VAR(z, order=1).fit(), n_frequencies=64).compute()
            >>> plain = density.coherence("y1", "y2")[1]
            >>> given = density.partial_coherence("y1", "y2")[1]
            >>> bool(plain > 0.3), bool(given < 0.1)
            (True, True)
        """
        i, j = self._index(first), self._index(second)
        inverse = np.linalg.inv(self.density)
        cross = np.abs(inverse[:, i, j]) ** 2
        power = np.real(inverse[:, i, i]) * np.real(inverse[:, j, j])
        return np.asarray(cross / np.maximum(power, 1e-300), dtype=np.float64)

    def gain(self, effect: str, cause: str) -> npt.NDArray[np.float64]:
        r"""The gain of the regression of ``effect`` on ``cause`` by frequency.

        .. math::

           G_{ij}(\omega) = \frac{|f_{ij}(\omega)|}{f_{jj}(\omega)},

        with :math:`i` the effect and :math:`j` the cause: the amplitude
        multiplier the best linear frequency-wise predictor of ``effect``
        from ``cause`` applies at each frequency.

        Args:
            effect: The variable being explained.
            cause: The variable explaining it.

        Returns:
            A real non-negative ``(n,)`` array.

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
            >>> density = SpectralDensity(res, n_frequencies=64).compute()
            >>> gain = density.gain("gdp", "rate")
            >>> gain.shape, bool(np.all(gain >= 0.0)), bool(gain[1] > gain[-1])
            ((64,), True, True)
        """
        i, j = self._index(effect), self._index(cause)
        return np.asarray(
            np.abs(self.density[:, i, j]) / np.maximum(np.real(self.density[:, j, j]), 1e-300),
            dtype=np.float64,
        )

    def phase(self, effect: str, cause: str) -> npt.NDArray[np.float64]:
        r"""The phase of the cross-spectrum ``f[effect, cause]``, in radians.

        :math:`\arg f_{ij}(\omega)` with :math:`i` the effect and
        :math:`j` the cause. Under the transfer convention
        :math:`e^{-i\omega l}` used throughout the package, a *negative*
        phase at frequency :math:`\omega` means ``cause`` leads
        ``effect`` by :math:`-\phi(\omega) / \omega` observations at that
        frequency: for ``effect_t = cause_{t-1}`` exactly, the phase is
        :math:`-\omega` on the whole grid. Half the literature states the
        mirror convention, so check signs against a known lead before
        trusting a plot.

        Args:
            effect: The variable being explained.
            cause: The variable explaining it.

        Returns:
            A real ``(n,)`` array in ``(-pi, pi]``.

        Raises:
            SpecificationError: If either label is unknown.

        Example:
            A one-period lead recovers the phase :math:`-\omega` exactly:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(1)
            >>> x = rng.standard_normal(2000)
            >>> y = np.zeros((2000, 2))
            >>> y[:, 1] = x
            >>> y[1:, 0] = x[:-1] + 0.05 * rng.standard_normal(1999)
            >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
            >>> density = SpectralDensity(res, n_frequencies=16).compute()
            >>> lead = density.phase("gdp", "rate")[1:] / density.frequencies[1:]
            >>> bool(np.allclose(lead, -1.0, atol=0.01))
            True
        """
        i, j = self._index(effect), self._index(cause)
        return np.asarray(np.angle(self.density[:, i, j]), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        r"""Build the structured summary: peak frequency and power share per variable.

        For each variable, the grid frequency of its spectral peak
        (excluding frequency zero), that frequency as a period, and its
        share of the system's total power,
        :math:`\int f_{ii} / \int \operatorname{tr} f`; the notes restate
        that the density is the model's, give the normalization, and say
        why zero is excluded from the peak search.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((500, 2))
            >>> for t in range(1, 500):
            ...     y[t, 0] = 0.5 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.standard_normal()
            ...     y[t, 1] = 0.5 * y[t - 1, 1] + rng.standard_normal()
            >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
            >>> table = SpectralDensity(res, n_frequencies=64).compute()._summary_table()
            >>> table.columns
            ('variable', 'peak frequency', 'peak period', 'power share')
            >>> table.rows[0]
            ('gdp', '0.0499', '126.0', '59.5%')
        """
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
    r"""The frequency-domain view of any fitted closed reduced form.

    Not an estimator: constructs with a fitted result -- anything exposing
    the closed-system surface -- and evaluates its exact spectral density
    matrix

    .. math::

        f(\omega) = \frac{1}{2\pi}\,\Psi(\omega)\,\Sigma_u\,\Psi(\omega)^*,
        \qquad
        \Psi(\omega) = \Bigl[I - \sum_{l=1}^{p} A_l e^{-i\omega l}\Bigr]^{-1},

    on an inclusive uniform grid of ``n_frequencies`` points on
    :math:`[0, \pi]`. Nothing is re-estimated and no data are read: the
    coefficient stack :math:`A_1, \dots, A_p` and the innovation covariance
    :math:`\Sigma_u` are the whole input, so the same closed form serves an
    OLS VAR, a VARMA through its autoregressive representation, an
    error-correction model in levels, or a sparse hundred-variable system.
    The rest of the surface -- spectra, coherence, gain, phase -- is read
    off the returned :class:`SpectralDensityResult`.

    Attributes:
        _result: The fitted closed-system result the density is read from.
        _n_frequencies: Grid points on :math:`[0, \pi]`, endpoints included.

    Args:
        result: A fitted closed reduced-form result: anything satisfying
            :class:`~cultivars._core.ClosedSystemResult`.
        n_frequencies: Grid points on :math:`[0, \pi]`, endpoints
            included; at least two.

    Raises:
        SpecificationError: If ``result`` does not expose the closed-system
            surface (an unfitted model, a univariate result, a result
            without a moving-average representation) or ``n_frequencies``
            is below two.

    Note:
        The density is the fitted model's, not the data's: every
        specification error the time-domain fit carries -- a lag order
        chosen short, a near-unit root the OLS fit left at 0.98 -- is
        inherited exactly. A near-unit root produces a legitimately
        enormous spectrum at frequency zero and is *not* an error; only an
        exact unit root, which makes the lag polynomial singular, raises,
        and it does so from :meth:`compute` rather than from construction.
        The grid is a choice of resolution, not of estimation: a business
        cycle band of periods 6 to 32 quarters spans
        :math:`\omega \in [0.196, 1.047]`, and at the default 256 points
        that band holds roughly 70 grid points.

    See Also:
        * :class:`SpectralDensityResult` -- the container this produces.
        * :class:`~cultivars.spectral.causality.SpectralCausality` -- the
          directional decomposition of the same matrix.
        * :class:`~cultivars.spectral.periodogram.SpectrumEstimate` -- the
          nonparametric estimate from the data, which this class
          deliberately does not stand in for.
        * :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
          -- the other time-domain-free view built on the same
          closed-system surface.

    References:
        Hamilton, J. D. (1994). *Time Series Analysis*, chapter 6. Princeton
        University Press.

        Priestley, M. B. (1981). *Spectral Analysis and Time Series*.
        Academic Press.

    Example:
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((400, 2))
        >>> for t in range(1, 400):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
        >>> spec = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=128).compute()
        >>> spec.density.shape
        (128, 2, 2)
        >>> bool(np.all(spec.coherence("y1", "y2") <= 1.0 + 1e-12))
        True

        For a univariate AR(1) the closed form is
        :math:`f(0) = \sigma^2 / (2\pi (1 - \phi)^2)`, and the view returns
        exactly that number at the first grid point:

        >>> fit = VAR(y[:, :1], order=1).fit()
        >>> phi, sigma2 = float(fit.coefficients[0, 0, 0]), float(fit.sigma_u[0, 0])
        >>> spec = SpectralDensity(fit, n_frequencies=16).compute()
        >>> bool(np.isclose(spec.spectrum("y1")[0], sigma2 / (2 * np.pi * (1 - phi) ** 2)))
        True
    """

    __slots__ = ("_n_frequencies", "_result")

    def __init__(self, result: ClosedSystemResult, *, n_frequencies: int = 256) -> None:
        """Validate the source surface and the grid.

        Args:
            result: A fitted closed reduced-form result.
            n_frequencies: Grid points on ``[0, pi]``, endpoints included.

        Raises:
            SpecificationError: If ``result`` does not satisfy the
                closed-system protocol or ``n_frequencies`` is below two.

        Example:
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((100, 2))
            >>> SpectralDensity(VAR(y, order=1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: SpectralDensity reads a fitted ... got VAR.
            >>> SpectralDensity(VAR(y, order=1).fit(), n_frequencies=1)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n_frequencies must be at least 2; got 1.
        """
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
        r"""Evaluate the exact density on the grid.

        Builds the grid, the transfer function :math:`\Psi(\omega)` at each
        point, and the Hermitian matrix
        :math:`\Psi(\omega)\Sigma_u\Psi(\omega)^*/2\pi`. The call is pure:
        repeated calls return equal, independent results.

        Returns:
            The :class:`SpectralDensityResult` on ``n_frequencies`` points
            from ``0`` to ``pi``.

        Raises:
            NumericalError: If the lag polynomial is exactly singular at
                some grid frequency (an exact unit root), or the density
                loses Hermitian symmetry.

        Example:
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> view = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=5)
            >>> spec = view.compute()
            >>> spec.frequencies.round(3).tolist()
            [0.0, 0.785, 1.571, 2.356, 3.142]
            >>> bool(np.array_equal(spec.density, view.compute().density))
            True
        """
        result = self._result
        grid = frequency_grid(self._n_frequencies)
        transfer = transfer_function(np.asarray(result.coefficients), grid)
        return SpectralDensityResult(
            names=tuple(result.names),
            frequencies=grid,
            density=spectral_matrix(transfer, np.asarray(result.sigma_u)),
        )
