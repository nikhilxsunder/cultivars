# filepath: /src/cultivars/forecast/scoring.py
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

"""Proper scores for density forecasts: what a predictive sample earned.

A density forecast is a claim about a distribution, and only a *strictly
proper* scoring rule -- one a forecaster cannot improve by hedging away
from their honest belief -- deserves to grade it (Gneiting & Raftery
2007). That is the admission criterion here. The CRPS grades each series
exactly from the predictive sample; the energy score is its multivariate
generalization, grading the joint claim; the log score is the classical
alternative, and the one number in this module that is *estimated* (by a
kernel density on the draws) rather than exact, which its docstring says
plainly. The point losses ride along as degenerate cases because backtests
report them, not because they suffice: a point forecast is a density
forecast that has thrown its uncertainty away.

This module carries a load the design placed deliberately: the Gibbs BVAR
family reports no marginal likelihood, so out-of-sample density scores are
the honest instrument that ranks a conjugate fit against a horseshoe fit
against anything else that produces predictive paths. Everything scores
raw arrays -- ``forecast_paths`` output, or any rival forecaster's
simulations -- so the comparison is never restricted to this package's
own models. All scores are negatively oriented (smaller is better), and
everything is reported per horizon and per series before any averaging,
because aggregation is a modelling choice the user should make in
daylight.

References:
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. *Journal of the American Statistical
        Association*, 102(477), 359-378.
    Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
        forecasts, calibration and sharpness. *Journal of the Royal
        Statistical Society B*, 69(2), 243-268.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, crps_from_draws, energy_score
from .._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DensityScoreResult(_SummaryMixin):
    """One forecast origin's scores, per horizon and per series.

    Every score is negatively oriented -- smaller is better -- and nothing
    is averaged: aggregation across horizons or series is the caller's
    explicit act, made in daylight.

    Attributes:
        names: One label per series.
        crps: ``(h, k)`` continuous ranked probability scores, exact from
            the predictive sample, in each variable's own units.
        log_score: ``(h, k)`` negative log predictive density at the
            outcome, the density estimated by a Gaussian kernel on the
            draws with Silverman's bandwidth -- the module's one estimated
            quantity, and the score most sensitive to tail misfit.
        energy: ``(h,)`` energy scores of the joint predictive, the
            multivariate CRPS.
        squared_error: ``(h, k)`` squared errors of the predictive mean.
        absolute_error: ``(h, k)`` absolute errors of the predictive
            median.
        n_draws: Predictive draws scored against.
    """

    names: tuple[str, ...]
    crps: npt.NDArray[np.float64] = field(repr=False)
    log_score: npt.NDArray[np.float64] = field(repr=False)
    energy: npt.NDArray[np.float64] = field(repr=False)
    squared_error: npt.NDArray[np.float64] = field(repr=False)
    absolute_error: npt.NDArray[np.float64] = field(repr=False)
    n_draws: int

    @property
    def n_horizons(self) -> int:
        """Horizons scored."""
        return int(self.crps.shape[0])

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = tuple(
            (
                name,
                f"{float(self.crps[:, index].mean()):.4f}",
                f"{float(self.log_score[:, index].mean()):.4f}",
                f"{float(np.sqrt(self.squared_error[:, index].mean())):.4f}",
            )
            for index, name in enumerate(self.names)
        )
        notes = [
            "All scores are negatively oriented: smaller is better. The "
            "CRPS and energy score are exact functionals of the predictive "
            "sample; the log score is kernel-estimated on the draws and is "
            "the number most sensitive to tail misfit.",
            f"Joint energy score, averaged over horizons: {float(self.energy.mean()):.4f}.",
            "Rows average over horizons for display only; the per-horizon "
            "arrays are the result, and aggregation is the caller's "
            "explicit choice.",
            "Scores are statements about realized samples, not posterior "
            "probabilities of model truth; nothing here is a Bayes factor.",
        ]
        return SummaryTable(
            title="Density Forecast Scores",
            metadata=(
                ("Series", f"{len(self.names)}"),
                ("Horizons", f"{self.n_horizons}"),
                ("Draws", f"{self.n_draws}"),
            ),
            columns=("series", "mean CRPS", "mean log score", "RMSE"),
            rows=rows,
            notes=tuple(notes),
        )


class DensityScore:
    """Score one origin's density forecast against what happened.

    Constructs from raw arrays -- the ``forecast_paths`` output of any
    Bayesian result in the package, or any rival forecaster's simulated
    paths -- so the grading is model-agnostic by design.

    Args:
        paths: ``(n_draws, h, k)`` predictive paths; a single-horizon
            ``(n_draws, k)`` block is promoted to ``h = 1``.
        realized: ``(h, k)`` outcomes over the same horizons; ``(k,)`` is
            promoted to ``h = 1``.
        names: One label per series. Defaults to ``y1 ... yk``.

    Raises:
        DimensionError: If the arrays cannot be scored together.
        NumericalError: If any input is not finite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> paths = rng.standard_normal((2000, 4, 2))
        >>> outcome = rng.standard_normal((4, 2))
        >>> scores = DensityScore(paths, outcome).compute()
        >>> scores.crps.shape
        (4, 2)
        >>> bool(np.all(scores.crps > 0.0))
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
        """Validate and align the sample and the outcomes."""
        block = np.asarray(paths, dtype=np.float64)
        if block.ndim == 2:
            block = block[:, None, :]
        outcome = np.asarray(realized, dtype=np.float64)
        if outcome.ndim == 1:
            outcome = outcome[None, :]
        if block.ndim != 3 or outcome.ndim != 2 or block.shape[1:] != outcome.shape:
            raise DimensionError(
                f"paths must be (n_draws, h, k) against (h, k) realizations; "
                f"got {np.asarray(paths).shape} against "
                f"{np.asarray(realized).shape}."
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

    def _kernel_log_score(self) -> npt.NDArray[np.float64]:
        """Negative log predictive density by Gaussian kernel, per cell."""
        draws, horizons, k = self._paths.shape
        out = np.empty((horizons, k))
        for h in range(horizons):
            block = self._paths[:, h, :]
            spread = block.std(axis=0, ddof=1)
            quartiles = np.quantile(block, [0.25, 0.75], axis=0)
            robust = (quartiles[1] - quartiles[0]) / 1.34
            width = 0.9 * np.minimum(spread, np.where(robust > 0.0, robust, spread))
            width = np.maximum(width * draws ** (-0.2), 1e-8 * np.maximum(spread, 1.0))
            gap = (block - self._realized[h][None, :]) / width[None, :]
            peak = (-0.5 * gap**2).max(axis=0)
            total = np.exp(-0.5 * gap**2 - peak[None, :]).sum(axis=0)
            out[h] = -(
                peak + np.log(total) - np.log(draws) - np.log(width) - 0.5 * np.log(2.0 * np.pi)
            )
        return out

    def compute(self) -> DensityScoreResult:
        """Evaluate every score.

        Returns:
            The :class:`DensityScoreResult`.
        """
        draws, horizons, k = self._paths.shape
        flat = self._paths.reshape(draws, horizons * k)
        crps = crps_from_draws(flat, self._realized.ravel()).reshape(horizons, k)
        energy = np.array(
            [energy_score(self._paths[:, h, :], self._realized[h]) for h in range(horizons)]
        )
        mean_error = self._paths.mean(axis=0) - self._realized
        median = np.median(self._paths, axis=0)
        return DensityScoreResult(
            names=self._names,
            crps=crps,
            log_score=self._kernel_log_score(),
            energy=energy,
            squared_error=mean_error**2,
            absolute_error=np.abs(median - self._realized),
            n_draws=draws,
        )
