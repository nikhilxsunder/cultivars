# filepath: /src/cultivars/_core/_defaults.py
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

"""Package-wide numeric constants and estimator defaults.

Every magic number in the package resolves here. Constants are private by
naming convention (leading underscore) because they are implementation
detail: changing one changes fitted output, so they are not public API.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

_SCHEMA_VERSION: int = 1
"""Serialization schema version stamped onto every result object."""

_LOG_2PI: float = float(np.log(2.0 * np.pi))
"""``log(2 * pi)``; the Gaussian log-likelihood constant."""

_SQRT_2_OVER_PI: float = float(np.sqrt(2.0 / np.pi))
"""``E|z|`` for standard normal ``z``; the EGARCH asymmetry centering term."""

_PACF_CLIP: float = 0.999
"""Clip applied to partial autocorrelations before the arctanh transform."""

_D_MAX: float = 0.499
"""Upper bound on the fractional differencing parameter (stationary region)."""

_PERSISTENCE_MAX: float = 0.999
"""Upper bound on total variance persistence for a covariance-stationary fit."""

_ROW_SUM_ATOL: float = 1e-6
"""Absolute tolerance when checking that transition-matrix rows sum to one."""

_STABILITY_TOL: float = 1e-8
"""Default modulus tolerance for unit-root and explosive-root classification."""

_TINY: float = 1e-300
"""Floor guarding logarithms of mixture densities against underflow."""

_DEFAULT_TRUNCATION: int = 1000
"""Default ARCH(infinity) / AR(infinity) truncation lag."""

_DEFAULT_GRID: int = 300
"""Default number of candidate thresholds in a threshold grid search."""

_DEFAULT_TRIM: float = 0.15
"""Default fraction trimmed from each tail of a threshold grid."""

_DEFAULT_MAX_ITER: int = 500
"""Default maximum EM iterations."""

_DEFAULT_TOL: float = 1e-6
"""Default relative convergence tolerance for iterative estimators."""

_DEFAULT_STARTS: int = 10
"""Default number of random restarts for multimodal likelihoods."""

_GPH_EXPONENT: float = 0.5
"""Default bandwidth exponent ``m = floor(n ** 0.5)`` for the GPH estimator."""

_WHITTLE_EXPONENT: float = 0.65
"""Default bandwidth exponent for the local Whittle estimator."""

_TREND_WIDTH: dict[str, int] = {"n": 0, "c": 1, "ct": 2}
"""Number of deterministic columns implied by each trend specification."""

_PENALTY: Final[float] = 1e10
"""Criterion value returned for an inadmissible or numerically failed draw.

Large enough that no admissible parameter vector competes with it, finite so
that a gradient-based optimizer can still step away from it.
"""

_EXTRA: dict[str, str] = {"pandas": "pandas", "polars": "polars"}
"""Distribution name for each optional frame backend, keyed by module name."""

_CAPACITY_WARNING: float = 0.1
"""Learner parameters per observation above which the summary flags capacity."""

_DEFAULT_ALPHA: Final[float] = 0.05
"""Significance level assumed by a bare ``reject()`` call and by the repr."""

_RANK_TOL: float = 1e-10
"""Tolerance for determining the numerical rank of a matrix."""

_KSC_PROB: npt.NDArray[np.float64] = np.array(
    [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750]
)
"""Mixture weights of the KSC seven-component log chi-squared approximation."""

_KSC_MEAN: npt.NDArray[np.float64] = (
    np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
)
"""Component means, already shifted by the log chi-squared mean of -1.2704."""

_KSC_VAR: npt.NDArray[np.float64] = np.array(
    [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261]
)
"""Component variances."""

_OFFSET = 1e-6
"""Offset inside ``log(e**2 + offset)``, guarding the log at exact zeros."""


_STUDENT_DF_GRID: npt.NDArray[np.float64] = np.concatenate(
    [np.arange(2.0, 30.0), np.array([30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0])]
)
"""Grid of candidate degrees of freedom for the Student-t innovations."""

_GIG_TINY = 1e-12
"""Parameter floor below which a GIG boundary face is drawn as its limit."""

_GIG_MAX_ROUNDS = 500
"""Vectorized rejection rounds before a conditional is declared degenerate."""

_LOG_CHI2_MEAN = -1.2704
"""Mean of ``log(eps**2)`` for standard-normal ``eps``."""

_LOG_CHI2_VAR = float(np.pi**2 / 2.0)
"""Variance of ``log(eps**2)`` for standard-normal ``eps``."""

_TARGET_ACCEPTANCE = 0.234
"""Roberts-Gelman-Gilks optimal random-walk acceptance rate, the tuning target."""

_ADAPT_FLOOR = 1e-8
"""Diagonal jitter added to the adapted covariance so it never degenerates."""

_NU_PRIOR_RATE = 0.1
"""Rate parameter for the prior on the degrees of freedom of the Student-t innovations."""

_MIN_CHAIN_DRAWS: Final[int] = 8
"""Fewest kept draws per chain a convergence diagnostic will accept.

Split-R-hat halves each chain and the Geyer sequence needs a few lags on each
half, so anything shorter has no autocorrelation structure to estimate from
and the statistic would be a number wearing the name of a diagnostic.
"""

_RHAT_TOL: Final[float] = 1.01
"""Rank-normalized split-R-hat above which a parameter is flagged.

Vehtari et al. (2021) tightened the classical 1.1 to 1.01 after showing that
chains with R-hat between the two can still disagree materially in their
tails.
"""

_MIN_ESS_PER_CHAIN: Final[float] = 100.0
"""Bulk and tail effective draws per chain below which a parameter is flagged.

The same paper's floor: enough draws that the Monte Carlo standard error of a
posterior quantile is small relative to its posterior spread.
"""

_MHM_TAU: Final[float] = 0.9
"""Probability mass of the Gaussian envelope Geweke's harmonic mean retains."""

_BRIDGE_TOL: Final[float] = 1e-10
"""Relative change in the log evidence at which the bridge recursion stops."""

_BRIDGE_MAX_ITER: Final[int] = 1000
"""Iterations of the bridge recursion before it is declared not to converge."""

_DISCREPANCY_STATISTICS: Final[tuple[str, ...]] = ("mean", "sd", "acf1", "kurtosis", "arch1")
"""Per-variable discrepancy statistics a predictive check computes by default.

Location, scale, first-order persistence, tail weight, and first-order
persistence of the squares: the five features a stationary time-series model
is most often asked to reproduce, and the ones whose failure names the
misspecification (a Gaussian innovation cannot match ``kurtosis``; a constant
covariance cannot match ``arch1``).
"""

_MIN_REPLICATIONS: Final[int] = 20
"""Fewest replicated data sets a predictive p-value is reported from.

Below this a tail probability has a resolution coarser than 0.05, and the
verdict would be a rounding artefact.
"""

_EXTREME_PVALUE: Final[float] = 0.05
"""Predictive p-value below which, or above one minus which, a statistic is flagged.

A posterior predictive p-value is not uniform under the true model -- it is
conservative, concentrated near one half (Meng, 1994) -- so a flag at this
level understates the evidence against the model rather than overstating it.
"""

_KSC_MU_PRIOR: Final[tuple[float, float]] = (0.0, 10.0)
"""Kim, Shephard and Chib's ``(mean, variance)`` of the Gaussian prior on the log-variance mean."""

_KSC_PHI_PRIOR: Final[tuple[float, float]] = (20.0, 1.5)
"""Kim, Shephard and Chib's ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.

Prior mean ``2a / (a + b) - 1 = 0.86``: persistent volatility, with mass kept
off the unit root so the stationary log variance is well defined.
"""

_KSC_SIGMA2_PRIOR: Final[tuple[float, float]] = (2.5, 0.025)
"""Kim, Shephard and Chib's ``(shape, rate)`` of the inverse-gamma prior on ``sigma2``.

Prior mean ``rate / (shape - 1) = 0.0167``, a volatility path that moves
slowly relative to the returns it scales.
"""

_UCSV_VOL_OF_VOL_PRIOR: Final[tuple[float, float]] = (3.0, 0.04)
"""``(shape, rate)`` of the inverse-gamma prior on each random-walk log-variance step variance.

Prior mean ``0.02``, close to Stock and Watson's (2007) fixed ``gamma**2``
and the value the unobserved-components model holds when ``gamma`` is stated.
"""

_MIN_COMPARISON_ORIGINS: Final[int] = 8
"""Fewest evaluation origins a forecast comparison test accepts.

Below this the long-run variance of a loss differential is estimated from
too few observations for either the Diebold-Mariano or the Clark-West
statistic to have usable size.
"""

_MCS_BOOTSTRAP: Final[int] = 1000
"""Stationary-bootstrap replications behind the model confidence set."""

_MCS_ALPHA: Final[float] = 0.1
"""Confidence level complement of the model confidence set.

Hansen, Lunde and Nason (2011) report the 90% set: at 10% the set is
informative on the short evaluation windows macroeconomic forecasting has,
where a 95% set routinely retains every model.
"""

_CRITICAL_LEVELS: Final[tuple[str, str, str]] = ("1%", "5%", "10%")
"""Labels of the three levels every tabulated critical-value record carries, in this order."""

_CRITICAL_ALPHAS: Final[tuple[float, float, float]] = (0.01, 0.05, 0.10)
"""The sizes behind ``_CRITICAL_LEVELS``, in the same order."""

_SIMULATION_BURN: Final[int] = 100
"""Periods a fresh simulated sample discards before the kept part begins.

Enough for a stationary autoregression of moderate persistence to forget a
zero start; a near-unit-root or long-memory law needs more, and every
``simulate`` takes ``burn`` explicitly.
"""

_HANSEN_REPLICATIONS: Final[int] = 1000
"""Fixed-regressor simulations behind Hansen's sup-F p-value."""

_BDS_RADIUS: Final[float] = 1.5
"""BDS radius in standard deviations of the series; the best-sized choice in Brock et al."""


_LW_BOUNDS: Final[tuple[float, float]] = (-0.5, 1.0)
"""Bounds for the local Whittle estimator."""

_ELW_BOUNDS: Final[tuple[float, float]] = (-0.5, 2.0)
"""Bounds for the exact local Whittle estimator."""

_METHOD_LABELS: Final[dict[str, str]] = {
    "gph": "GPH",
    "local_whittle": "Local Whittle",
    "exact_local_whittle": "Exact Local Whittle",
}
"""Labels for the long-memory estimation methods."""

_MIN_LONG_MEMORY_OBS: Final[int] = 64
"""Minimum number of observations required for long-memory estimation."""

_HEGY_REPLICATIONS: Final[int] = 2000
"""Seasonal random walks simulated behind the HEGY p-values."""

_MIN_SEASONAL_CYCLES: Final[int] = 6
"""Minimum number of seasonal cycles required for a seasonal unit-root test."""

_METHOD_TITLES: Final[dict[str, str]] = {
    "hodrick-prescott": "Hodrick-Prescott",
    "butterworth": "Butterworth",
    "hamilton": "Hamilton",
    "beveridge-nelson": "Beveridge-Nelson",
}
"""Titles for the trend-cycle decomposition methods."""

_MIN_SPECTRUM_OBS: Final[int] = 32
"""Minimum number of observations required for spectral estimation."""

_SPECTRAL_METHOD_TITLES: Final[dict[str, str]] = {
    "daniell": "Smoothed Periodogram",
    "multitaper": "Multitaper",
    "welch": "Welch",
}
"""Titles for the spectral estimation methods."""
