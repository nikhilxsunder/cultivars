# filepath: /examples/plot_04_heteroskedasticity_and_proxy.py
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
Monetary policy without recursiveness: heteroskedasticity and proxies
=====================================================================

Recursive identification of a monetary policy shock rests on a timing
assumption that nobody believes literally. Two alternatives drop it.
Rigobon (2003) identifies from a *shift in volatility*: if the impact
matrix is constant while the shock variances change across declared
regimes, the regime covariances pin the impact matrix down with no
zeros at all. Mertens and Ravn (2013) identify from an *external
instrument*: a series correlated with the policy shock and with nothing
else identifies the policy column by two-stage least squares on the
reduced-form innovations.

The two combine naturally. The Volcker disinflation of 1979-82, when
the Fed targeted reserves and let the funds rate move freely, is the
canonical variance regime for a monetary shock, and the shock series it
recovers in a small system can serve as the proxy in a larger one --
the two-step design of Lanne and Lütkepohl (2008). This example runs
that design and compares the three sets of impulse responses.
"""

# %%
# Data
# ----
# Monthly 1965-2007: industrial production, the CPI, commodity prices,
# the funds rate and M2, in the units of the recursive example. The
# small system uses the first, second and fourth; the large system
# uses all five over the same sample and with the same lag order, so
# that the residual rows of the two VARs line up date for date.

import matplotlib.pyplot as plt
import numpy as np

from cultivars.multivariate.reduced_form.vector_autoregression import VAR
from cultivars.multivariate.structural.external_instruments import ProxySVAR
from cultivars.multivariate.structural.heteroskedacity import HeteroskedasticSVAR
from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

from _fred import fetch

data = fetch(
    {"ip": "INDPRO", "cpi": "CPIAUCSL", "pcom": "PPIACO", "ffr": "FEDFUNDS", "m2": "M2SL"},
    start="1965-01-01",
    end="2007-12-01",
    units={"ip": "log", "cpi": "log", "pcom": "log", "m2": "log"},
)
data[["ip", "cpi", "pcom", "m2"]] *= 100.0
order = 12
small_names = ("ip", "cpi", "ffr")
small = VAR(data[list(small_names)].to_numpy(), order=order, names=small_names).fit()
dates = data.index[order:]

# %%
# The variance regime
# -------------------
# One label per residual row. The Volcker regime runs from October 1979,
# when the FOMC switched to non-borrowed-reserve targeting, to October
# 1982, when it abandoned it. Everything else is the calm regime; it is
# listed first, so the shocks are normalized to unit variance there and
# the reported ratios read as "how many times more volatile in 1979-82".

volcker = (dates >= "1979-10-01") & (dates <= "1982-10-01")
regimes = np.where(volcker, "volcker", "calm")
print(f"{int(volcker.sum())} Volcker months, {int((~volcker).sum())} calm months")

# %%
# Identification through heteroskedasticity
# -----------------------------------------
# With two regimes the rotation comes from an eigendecomposition and is
# unique when the variance ratios are distinct. The summary reports
# those ratios; the shock whose variance shifted most is in the first
# column. Nothing in the data labels it "monetary", so the check is
# whether that shock's impact pattern is the one a policy shock should
# have: it moves the funds rate on impact, by much more than it moves
# production or prices.

rigobon = HeteroskedasticSVAR(small, regimes, shock_names=("policy", "second", "third")).identify()
print(rigobon.summary())

horizon = 48
ffr = small_names.index("ffr")
policy = list(rigobon.shock_names).index("policy")
irf_rigobon = rigobon.irf(horizon)[:, :, policy]
irf_rigobon = irf_rigobon / irf_rigobon[0, ffr]

# %%
# The recursive benchmark
# -----------------------
# The same small system under the Cholesky ordering with the rate last,
# normalized the same way, is the comparison the literature always draws.

recursive = RecursiveSVAR(small, order=small_names).identify()
irf_recursive = recursive.irf(horizon)[:, :, ffr]
irf_recursive = irf_recursive / irf_recursive[0, ffr]

# %%
# Correlation of the two shock series
# -----------------------------------
# If both schemes identify the same object the recovered shock series
# should be highly correlated. A low correlation means at least one of
# the identifying assumptions is wrong; it does not say which.

shock_rigobon = rigobon.structural_shocks()[:, policy]
shock_recursive = recursive.structural_shocks()[:, ffr]
correlation = np.corrcoef(shock_rigobon, shock_recursive)[0, 1]
print(f"Correlation of the two policy shock series: {correlation:.3f}")

# %%
# Second step: the Rigobon shock as an external instrument
# --------------------------------------------------------
# The shock recovered from the three-variable system is used as the
# proxy for the policy shock in a five-variable system that adds
# commodity prices and M2. The proxy is exogenous by construction with
# respect to the *small* system; its relevance in the large system is
# an empirical matter that the first-stage F statistic reports. The F
# statistic is the instrument's whole claim to usefulness: below ten
# the identified column is unreliable (Montiel Olea, Stock and Watson
# 2021).

large_names = tuple(data.columns)
large = VAR(data.to_numpy(), order=order, names=large_names).fit()
proxy = ProxySVAR(large, shock_rigobon, shock="policy", normalize="ffr").identify()
print(proxy.summary())
ffr_large = large_names.index("ffr")
irf_proxy = proxy.irf(horizon)[:, :, list(proxy.shock_names).index("policy")]
irf_proxy = irf_proxy / irf_proxy[0, ffr_large]

# %%
# Three readings of the same shock
# --------------------------------
# All three are normalized to a one-point impact increase in the funds
# rate. The recursive and heteroskedasticity responses come from the
# same three-variable system; the proxy responses come from the larger
# one, so differences there mix identification with specification.

labels = {"ip": "Industrial production (%)", "cpi": "Consumer prices (%)", "ffr": "Funds rate (pp)"}
steps = np.arange(horizon + 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, name in zip(axes, small_names, strict=True):
    ax.plot(steps, irf_recursive[:, small_names.index(name)], lw=2, label="recursive")
    ax.plot(steps, irf_rigobon[:, small_names.index(name)], lw=2, label="heteroskedasticity")
    ax.plot(steps, irf_proxy[:, large_names.index(name)], lw=2, ls="--", label="proxy (5-var)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title(labels[name])
    ax.set_xlabel("months after shock")
axes[0].legend(fontsize=8)
fig.suptitle("Responses to a one-point funds-rate shock under three schemes")
fig.tight_layout()

# %%
# What to take away
# -----------------
# Heteroskedasticity identification is only as good as the regime
# declaration and the constant-impact assumption, and the variance
# ratios in the summary are its entire empirical content: ratios close
# to one for all but one shock mean the scheme identifies that shock
# well and the others barely at all. The proxy step inherits whatever
# mislabelling the first step made -- a generated instrument is still an
# instrument, but its exogeneity is a claim about the small system, not
# a fact about the Fed. Where the three responses agree, the recursive
# timing assumption was not doing the work; where they diverge, it was.
