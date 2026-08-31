# filepath: /src/cultivars/multivariate/structural/external_instruments.py
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

"""External-instrument identification: one shock, named from outside the system.

An instrument correlated with one structural shock and uncorrelated with the
rest identifies that shock's impact column -- and nothing else, which is
exactly what the result carries. Narrative tax changes, high-frequency
monetary surprises, oil supply disruptions: the instrument brings information
the reduced form never had, and in exchange identifies only the shock it
speaks for. The partial result is the honest one, and the shared
:class:`~cultivars.multivariate.structural.SVARResult` surface is built so a
single identified column supports impulse responses, variance shares, shock
recovery, and historical contributions without fabricating the columns it
does not have.

References:
    Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and
        corporate income tax changes in the United States. *American Economic
        Review*, 103(4), 1212-1247.
    Stock, J. H., & Watson, M. W. (2018). Identification and estimation of
        dynamic causal effects in macroeconomics using external instruments.
        *Economic Journal*, 128(610), 917-948.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..._core import ClosedSystemResult
from ..._internals import _IdentificationModel
from ...exceptions import DimensionError, NumericalError, SpecificationError
from .zero_restrictions import SVARResult


class ProxySVAR(_IdentificationModel[SVARResult]):
    """External-instrument identification, Mertens-Ravn (2013).

    The identified column is recovered by two-stage least squares on the
    reduced-form innovations: the instrument's covariance with each innovation
    is proportional to the column, the ratio to the normalization variable
    removes the unknown instrument strength, and the unit-variance rescaling
    pins the scale against the innovation covariance.

    Both instrument conditions are assumptions, not testable implications:
    relevance shows up in the first-stage statistic the summary reports,
    exogeneity never shows up anywhere and must be argued from how the
    instrument was built.

    Args:
        result: The fitted closed reduced-form result to identify.
        instruments: ``(nobs, m)`` instrument panel aligned with the
            *effective* sample -- one row per residual row of the result,
            which is the estimation sample minus the burned lags.
        shock: Label for the identified shock.
        normalize: Variable whose impact response is normalized positive, and
            whose innovation anchors the first stage. Defaults to the first
            variable.

    Raises:
        SpecificationError: If the result is not a closed system, or the
            normalization variable is unknown.
        DimensionError: If the instrument panel does not align with the
            effective sample.
        NumericalError: If the instrument panel is non-finite.
    """

    __slots__ = ("_instruments", "_pivot", "_shock")

    def __init__(
        self,
        result: ClosedSystemResult,
        instruments: npt.ArrayLike,
        *,
        shock: str = "proxied",
        normalize: str | None = None,
    ) -> None:
        """Validate the source system, the instrument panel, and the anchor."""
        super().__init__(result)
        n = int(np.asarray(result.resid).shape[0])
        z = np.asarray(instruments, dtype=np.float64)
        if z.ndim == 1:
            z = z[:, np.newaxis]
        if z.ndim != 2 or z.shape[0] != n:
            raise DimensionError(
                f"instruments must have one row per residual row ({n}), the "
                f"effective sample after the burned lags; got shape {z.shape}."
            )
        if not np.all(np.isfinite(z)):
            raise NumericalError("instruments must be finite.")
        self._instruments = z - z.mean(axis=0)
        anchor = self.names[0] if normalize is None else str(normalize)
        if anchor not in self.names:
            raise SpecificationError(
                f"unknown normalization variable {anchor!r}; expected one of "
                f"{self.names}."
            )
        self._pivot = self.names.index(anchor)
        self._shock = str(shock)

    @property
    def k_instruments(self) -> int:
        """Number of instrument series."""
        return int(self._instruments.shape[1])

    def identify(self) -> SVARResult:
        """Recover the proxied shock's impact column.

        Returns:
            A structural result carrying the one identified column.

        Raises:
            NumericalError: If the instrument has no first-stage relationship
                with the normalization innovation.
        """
        resid = np.asarray(self.source.resid, dtype=np.float64)
        n = resid.shape[0]
        z = self._instruments
        m = self.k_instruments
        anchor = self.names[self._pivot]

        beta: npt.NDArray[np.float64] = np.linalg.lstsq(z, resid[:, self._pivot], rcond=None)[0]
        fitted = z @ beta
        explained = float(fitted @ fitted)
        residual = float(resid[:, self._pivot] @ resid[:, self._pivot]) - explained
        if explained <= 0.0 or n <= m:
            raise NumericalError(
                f"the instrument has no first-stage relationship with the "
                f"{anchor!r} innovation; a proxy that does not move the "
                "normalization variable identifies nothing."
            )
        f_stat = (explained / m) / (residual / (n - m))

        relative = resid.T @ fitted / explained
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        scale = float(relative @ np.linalg.solve(sigma, relative))
        column = (relative / np.sqrt(scale))[:, np.newaxis]

        return SVARResult(
            source=self.source,
            impact=column,
            shock_names=(self._shock,),
            scheme="proxy",
            restriction=(
                f"External instrument: {m} proxy series assumed correlated "
                f"with the {self._shock!r} shock and uncorrelated with every "
                "other structural shock. Relevance is reported below; "
                "exogeneity is an assumption the instrument's construction "
                "must defend, because no statistic here can."
            ),
            diagnostics=(
                ("First-stage F", f"{f_stat:.2f}"),
                ("Instruments", f"{m}"),
                ("Normalization", f"{anchor} > 0 on impact"),
            ),
        )