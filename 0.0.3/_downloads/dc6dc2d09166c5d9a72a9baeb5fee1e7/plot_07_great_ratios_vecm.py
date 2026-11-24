# filepath: /examples/plot_07_great_ratios_vecm.py
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
r"""
The great ratios: cointegration and a vector error-correction model
===================================================================

Balanced growth implies that consumption, investment and output share
one stochastic trend: the consumption-output and investment-output
ratios are stationary while each series is integrated. King, Plosser,
Stock and Watson (1991) made that the test of the real-business-cycle
model's long-run implications, with the Johansen procedure as the
instrument. Three I(1) series with one common trend means a
cointegrating rank of two.

This example runs the full chain: integration tests on levels and
differences, lag-order selection, the Johansen rank test with
simulated p-values, estimation of the VECM at the selected rank, the
normalized cointegrating vectors against their balanced-growth values,
weak-exogeneity tests of the adjustment, and a forecast that respects
the long-run restrictions.
"""

# %%
# Data
# ----
# Real personal consumption expenditures, real gross private domestic
# investment and real GDP, quarterly 1948-2019, as 100 times their
# logs so that cointegrating residuals read in percent.

import matplotlib.pyplot as plt
import numpy as np

from cultivars.diagnostics.unit_roots import adf, kpss
from cultivars.multivariate.reduced_form.error_correction import VECM

from _fred import fetch

levels = fetch(
    {"c": "PCECC96", "i": "GPDIC1", "y": "GDPC1"},
    start="1948-01-01",
    end="2019-12-01",
    frequency="q",
    aggregation_method="avg",
    units="log",
)
levels *= 100.0
names = tuple(levels.columns)
endog = levels.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(levels.index, levels["c"] - levels["y"], label="c - y")
axes[0].plot(levels.index, levels["i"] - levels["y"], label="i - y")
axes[0].set_title("Log ratios to output (x100)")
axes[0].legend()
axes[1].plot(levels.index, levels)
axes[1].set_title("Log levels (x100)")
axes[1].legend(names)
fig.tight_layout()

# %%
# Orders of integration
# ---------------------
# Each series should fail to reject a unit root in levels (ADF with a
# trend) and reject it in differences; KPSS from the other side. The
# two ratios should look stationary, which is the balanced-growth
# claim in its simplest form.

header = ("ADF level", "ADF diff", "KPSS level", "KPSS diff")
print(f"{'series':8s}  " + "  ".join(f"{h:>10s}" for h in header))
for name in names:
    series = levels[name].to_numpy()
    cells = (
        adf(series, trend="ct").pvalue,
        adf(np.diff(series), trend="c").pvalue,
        kpss(series, trend="ct").statistic,
        kpss(np.diff(series), trend="c").statistic,
    )
    print(f"{name:8s}  " + "  ".join(f"{cell:10.3f}" for cell in cells))
for label, ratio in (("c - y", levels["c"] - levels["y"]), ("i - y", levels["i"] - levels["y"])):
    test = adf(ratio.to_numpy(), trend="c")
    print(f"{label:10s}  ADF {test.statistic:7.3f}  p = {test.pvalue:.3f}")

# %%
# Lag order and the rank test
# ---------------------------
# The lag order is that of the VAR in levels; the VECM uses one fewer
# lag in differences. The rank test is Johansen's trace and maximum
# eigenvalue sequences, with p-values simulated for the deterministic
# case in hand. A constant restricted to the cointegrating space is the
# right case here: the ratios have non-zero means, the growth rates
# have drift.

candidate = VECM(endog, order=2, rank=1, names=names)
print(candidate.lag_order_selection(max_lags=8).to_table())
order = 2
rank_test = VECM(endog, order=order, rank=1, names=names).rank_test(seed=7)
print(rank_test.to_table())
rank = rank_test.selected_rank(alpha=0.05)
by_max = rank_test.selected_rank(alpha=0.05, statistic="max_eigenvalue")
print(f"Selected rank at 5%: trace {rank}, max-eigenvalue {by_max}")

# %%
# Estimation at rank two
# ----------------------
# Balanced growth predicts rank two whatever the test selects in a
# given sample, and the model below is estimated at that rank so the
# cointegrating vectors can be read against theory. Normalizing both
# vectors on output turns them into the two ratios; the balanced-growth
# values are (-1, 0, 1) and (0, -1, 1).

vecm = VECM(endog, order=order, rank=2, names=names).fit()
print(vecm.summary())
beta = vecm.normalized_beta(on=names.index("y"))
alpha = vecm.normalized_alpha(on=names.index("y"))
print("Cointegrating vectors normalized on y (rows c, i, y):")
print(np.round(beta, 3))
print("Adjustment coefficients on that basis:")
print(np.round(alpha, 4))

# %%
# Who adjusts?
# ------------
# Weak exogeneity of a variable means its equation carries no
# error-correction term: it drives the common trend without responding
# to disequilibrium. In the King-Plosser-Stock-Watson reading output is
# the candidate; the tests say which of the three, if any, the data
# allow to be treated that way.

for name in names:
    test = vecm.weak_exogeneity(name)
    print(f"{name}: Wald {test.statistic:7.3f}  df {test.df}  p = {test.pvalue:.3f}")

# %%
# The error-correction terms
# --------------------------
# The two terms are the deviations from the estimated long-run
# relations. Mean-reverting, persistent, and with recessions visible as
# investment falling below its long-run share.

terms = vecm.error_correction_terms()
dates = levels.index[: terms.shape[0]]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(dates, terms[:, 0] - terms[:, 0].mean(), label="ec1")
ax.plot(dates, terms[:, 1] - terms[:, 1].mean(), label="ec2")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_title("Error-correction terms (demeaned)")
ax.legend()
fig.tight_layout()

# %%
# Forecasts that respect the long run
# -----------------------------------
# A VECM forecast keeps the ratios tied to their long-run values; a VAR
# in differences would let them wander. The eight-quarter forecast
# below is in levels, with the common trend extrapolated by its drift.

steps = 8
forecast = vecm.forecast(steps)
horizon = np.arange(1, steps + 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (j, name) in zip(axes, enumerate(names), strict=True):
    ax.plot(np.arange(-19, 1), endog[-20:, j], color="k", lw=1.2, label="observed")
    ax.plot(horizon, forecast[:, j], color="C3", lw=2, label="forecast")
    ax.set_title(name)
    ax.set_xlabel("quarters from end of sample")
axes[0].legend()
fig.tight_layout()

# %%
# What to take away
# -----------------
# The sequential rank test is a procedure, not a test: its size is not
# the 5 percent of any row, and the economics should decide between
# adjacent ranks when the statistics are close. Reading the normalized
# vectors against (-1, 0, 1) and (0, -1, 1) is the substantive check;
# the gap between the estimated investment share and its theoretical
# value is a known finding from the post-1990 sample and a reason to
# re-run the exercise ending in 1988, as the original paper did.
