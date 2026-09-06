# filepath: /src/cultivars/forecast/calibration.py
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

"""Calibration: does reality land where the densities said it would?

A density forecast can *score* well and still lie about its own
uncertainty -- too narrow in the tails, biased in the center -- and the
probability integral transform is how that shows: feed each realization
through its own predictive distribution, and a calibrated forecaster
produces uniform draws. The shape of the departure is the diagnosis. A
U-shaped PIT histogram means overconfidence (reality keeps landing in the
tails the forecaster ruled out), a hump means overdispersion, a slope
means bias. Berkowitz's (2001) test makes the reading formal on the
normal-quantile scale, where uniformity plus independence becomes a
three-parameter Gaussian null with an exact likelihood ratio.

The input grammar differs from scoring's on purpose: calibration is a
statement across *many* evaluation origins, so it takes the rolling stack
-- one predictive sample per origin, one realization per origin -- that
the caller's driver loop produces. One caveat is stated rather than
hidden: the Berkowitz null assumes the transforms are one-step objects;
multi-step forecasts overlap mechanically, and their PIT series carry
serial correlation the test will read as miscalibration.

References:
    Berkowitz, J. (2001). Testing density forecasts, with applications to
        risk management. *Journal of Business & Economic Statistics*,
        19(4), 465-474.
    Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating
        density forecasts with applications to financial risk management.
        *International Economic Review*, 39(4), 863-883.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.stats as sst

from .._core import SummaryTable, pit_from_draws
from .._internals import _LikelihoodRatioTest, _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["Calibration", "CalibrationResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class CalibrationResult(_SummaryMixin):
    """The PIT series and its formal reading, per series.

    Attributes:
        names: One label per series.
        pit: ``(T, k)`` probability integral transforms, one per
            evaluation origin, interior to ``(0, 1)``. Uniform under
            correct calibration.
        tests: One Berkowitz likelihood-ratio verdict per series, in
            ``names`` order.
    """

    names: tuple[str, ...]
    pit: npt.NDArray[np.float64] = field(repr=False)
    tests: tuple[_LikelihoodRatioTest, ...]

    @property
    def n_origins(self) -> int:
        """Evaluation origins."""
        return int(self.pit.shape[0])

    def _index(self, name: str) -> int:
        """Resolve a series label.

        Raises:
            SpecificationError: If the label is unknown.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown series {name!r}; the panel has {self.names}."
            ) from None

    def berkowitz(self, name: str) -> _LikelihoodRatioTest:
        """One series' Berkowitz verdict.

        Args:
            name: A series label.

        Returns:
            The likelihood-ratio test record.
        """
        return self.tests[self._index(name)]

    def histogram(self, bins: int = 10) -> npt.NDArray[np.float64]:
        """PIT histogram counts on equal-width bins over ``(0, 1)``.

        The picture the diagnosis reads: flat is calibrated, U-shaped is
        overconfident, hump-shaped is overdispersed, sloped is biased.

        Args:
            bins: Number of equal-width bins, at least 2.

        Returns:
            A ``(bins, k)`` array of counts.

        Raises:
            SpecificationError: If fewer than two bins are asked for.
        """
        if bins < 2:
            raise SpecificationError(f"bins must be at least 2; got {bins}.")
        edges = np.linspace(0.0, 1.0, bins + 1)
        out = np.empty((bins, len(self.names)))
        for index in range(len(self.names)):
            out[:, index] = np.histogram(self.pit[:, index], bins=edges)[0]
        return out

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = tuple(
            (
                name,
                f"{float(self.pit[:, index].mean()):.3f}",
                f"{float(self.pit[:, index].std(ddof=0)):.3f}",
                f"{self.tests[index].pvalue:.4f}",
            )
            for index, name in enumerate(self.names)
        )
        notes = [
            "Under correct calibration the PIT series is uniform: mean "
            "0.500, standard deviation 0.289, Berkowitz p-value well away "
            "from zero. A U-shaped histogram is overconfidence, a hump is "
            "overdispersion, a slope is bias -- histogram() draws the "
            "picture.",
            "The Berkowitz null assumes one-step transforms; multi-step "
            "forecasts overlap mechanically, and their serial correlation "
            "reads as miscalibration here.",
        ]
        return SummaryTable(
            title="Density Forecast Calibration",
            metadata=(
                ("Series", f"{len(self.names)}"),
                ("Origins", f"{self.n_origins}"),
            ),
            columns=("series", "PIT mean", "PIT sd", "Berkowitz p"),
            rows=rows,
            notes=tuple(notes),
        )


class Calibration:
    """Read a rolling forecast record's calibration.

    Args:
        paths: ``(T, n_draws, k)`` predictive samples, one block per
            evaluation origin; ``(T, n_draws)`` is promoted to ``k = 1``.
        realized: ``(T, k)`` outcomes, aligned origin by origin; ``(T,)``
            is promoted to ``k = 1``.
        names: One label per series. Defaults to ``y1 ... yk``.

    Raises:
        DimensionError: If the stacks cannot be aligned.
        SpecificationError: If the record is too short for the Berkowitz
            null to mean anything.
        NumericalError: If any input is not finite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> paths = rng.standard_normal((80, 500, 2))
        >>> outcomes = rng.standard_normal((80, 2))
        >>> record = Calibration(paths, outcomes).compute()
        >>> record.pit.shape
        (80, 2)
        >>> bool(record.berkowitz("y1").pvalue > 0.01)
        True
    """

    __slots__ = ("_names", "_paths", "_realized")

    def __init__(
        self,
        paths: npt.ArrayLike,
        realized: npt.ArrayLike,
        *,
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate and align the rolling stacks."""
        block = np.asarray(paths, dtype=np.float64)
        if block.ndim == 2:
            block = block[:, :, None]
        outcome = np.asarray(realized, dtype=np.float64)
        if outcome.ndim == 1:
            outcome = outcome[:, None]
        if (
            block.ndim != 3
            or outcome.ndim != 2
            or block.shape[0] != outcome.shape[0]
            or block.shape[2] != outcome.shape[1]
        ):
            raise DimensionError(
                f"paths must be (T, n_draws, k) against (T, k) realizations; "
                f"got {np.asarray(paths).shape} against "
                f"{np.asarray(realized).shape}."
            )
        if block.shape[0] < 10:
            raise SpecificationError(
                f"a calibration record over {block.shape[0]} origins cannot "
                "support the Berkowitz test; provide at least 10."
            )
        if not (np.all(np.isfinite(block)) and np.all(np.isfinite(outcome))):
            raise NumericalError("paths and realizations must be finite.")
        k = block.shape[2]
        if names is None:
            resolved = tuple(f"y{index + 1}" for index in range(k))
        else:
            resolved = tuple(str(name) for name in names)
            if len(resolved) != k:
                raise DimensionError(
                    f"names must have one entry per series ({k}); got {len(resolved)}."
                )
        self._paths = block
        self._realized = outcome
        self._names = resolved

    def compute(self) -> CalibrationResult:
        """Transform every origin and test every series.

        Returns:
            The :class:`CalibrationResult`.
        """
        origins, _, k = self._paths.shape
        pit = np.empty((origins, k))
        for origin in range(origins):
            pit[origin] = pit_from_draws(self._paths[origin], self._realized[origin])
        transformed = np.asarray(sst.norm.ppf(pit), dtype=np.float64)
        tests = tuple(
            _LikelihoodRatioTest._berkowitz_test(transformed[:, index]) for index in range(k)
        )
        return CalibrationResult(names=self._names, pit=pit, tests=tests)
