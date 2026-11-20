# filepath: /examples/plot_06_business_cycle_regimes.py
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
Business-cycle regimes: Markov switching against the NBER chronology
====================================================================

Hamilton (1989) modelled quarterly GNP growth as an autoregression
whose intercept switches between two unobserved regimes governed by a
Markov chain, and found that the smoothed regime probabilities trace
the NBER recession dates with no business-cycle information supplied
to the model. The exercise has become the standard test of a
regime-switching estimator, and the NBER chronology is on FRED, so it
can be run with no hand-entered dates.

This example fits Hamilton's specification and two extensions to real
GDP growth, compares them by likelihood and by their agreement with
the NBER indicator, and then estimates a two-variable Markov-switching
VAR on monthly data where the recession regime shows up in the
covariance as much as in the mean.
"""

# %%
# Data
# ----
# Quarterly real GDP growth, continuously compounded and in percent,
# and the NBER recession indicator ``USRECQ``, 1952 to 2019. Hamilton's
# sample ended in 1984; the longer sample includes the Great Moderation,
# which is what motivates letting the variance switch.

import matplotlib.pyplot as plt
import numpy as np

from cultivars.multivariate.regime_switching.markov_switching import MarkovSwitchingVAR
from cultivars.univariate.regime_switching import MarkovSwitchingAR

from _fred import fetch

quarterly = fetch(
    {"growth": "GDPC1", "nber": "USRECQ"},
    start="1951-10-01",
    end="2019-12-01",
    frequency="q",
    aggregation_method="avg",
    units={"growth": "cch"},
)
growth = quarterly["growth"].to_numpy()
nber = quarterly["nber"].to_numpy() > 0.5
print(f"{len(growth)} quarters, {int(nber.sum())} of them NBER recession quarters")

# %%
# Three specifications
# --------------------
# Hamilton's MSI(2)-AR(4): the intercept switches, the autoregressive
# coefficients and the variance do not. The MSIH variant lets the
# variance switch as well; the MSIAH variant lets everything switch.
# All three are estimated by EM from several random starts, and the
# regimes are labelled by ascending intercept, so regime 0 is always
# the low-growth state.

order = 4
hamilton = MarkovSwitchingAR(
    growth, order=order, switching_mean=True, switching_variance=False
).fit(seed=1)
msih = MarkovSwitchingAR(growth, order=order, switching_mean=True, switching_variance=True).fit(
    seed=1
)
msiah = MarkovSwitchingAR(
    growth, order=order, switching_mean=True, switching_variance=True, switching_ar=True
).fit(seed=1)
print(hamilton.summary())
print(hamilton.compare(msih, msiah, criterion="bic"))

# %%
# A likelihood-ratio test of the variance restriction
# ---------------------------------------------------
# The number of regimes cannot be tested by a likelihood ratio: under
# the null the transition probabilities are unidentified (the Davies
# problem). The equal-variance restriction, with the regime count held
# fixed, is a regular nested hypothesis and the test is valid.

print(hamilton.likelihood_ratio_test(msih).summary())

# %%
# Regime characteristics
# ----------------------
# The regime table reports each regime's intercept and variance, its
# ergodic probability, its expected duration in quarters and the share
# of the sample the smoothed probabilities assign to it. Hamilton's
# estimates were a recession lasting about four quarters and an
# expansion lasting about ten.

print(msih.regime_table())
print(msih.transition_table())

# %%
# Agreement with the NBER chronology
# ----------------------------------
# The smoothed probability of regime 0 is compared with the NBER
# indicator, aligned on the effective sample. Two summaries: the
# quadratic probability score, the Brier score of the regime
# probability as a recession forecast, and the concordance of the
# most-likely regime with the chronology.

effective = quarterly.index[order:]
nber_effective = nber[order:]


def agreement(result: object, label: str) -> None:
    """Print quadratic probability score and concordance against NBER."""
    probability = result.probabilities("smoothed")[:, 0]
    qps = float(np.mean((probability - nber_effective) ** 2))
    concordance = float(np.mean((probability > 0.5) == nber_effective))
    print(f"{label:10s}  QPS {qps:.3f}   concordance {100 * concordance:5.1f}%")


for label, result in (("MSI", hamilton), ("MSIH", msih), ("MSIAH", msiah)):
    agreement(result, label)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(effective, 0, 1, where=nber_effective, color="grey", alpha=0.3, label="NBER")
ax.plot(effective, hamilton.probabilities("smoothed")[:, 0], color="C0", lw=1.2, label="MSI")
ax.plot(effective, msih.probabilities("smoothed")[:, 0], color="C3", lw=1.2, label="MSIH")
ax.set_ylim(0, 1)
ax.set_title("Smoothed probability of the low-growth regime")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()

# %%
# Filtered versus smoothed
# ------------------------
# The smoothed probabilities use the whole sample and are what a
# historian wants; the filtered probabilities use data up to each date
# and are what a forecaster had. The gap between them at the start of
# each recession is the recognition lag of a real-time regime call.

filtered = msih.probabilities("filtered")[:, 0]
smoothed = msih.probabilities("smoothed")[:, 0]
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.fill_between(effective, 0, 1, where=nber_effective, color="grey", alpha=0.3)
ax.plot(effective, filtered, color="C1", lw=1.0, label="filtered")
ax.plot(effective, smoothed, color="C3", lw=1.2, label="smoothed")
ax.set_ylim(0, 1)
ax.set_title("MSIH: filtered and smoothed recession probabilities")
ax.legend(loc="upper right")
fig.tight_layout()

# %%
# A Markov-switching VAR on monthly data
# --------------------------------------
# Industrial production growth and the change in the unemployment rate,
# monthly 1960-2019, with the intercepts and the innovation covariance
# switching. The recession regime in a bivariate system is a state of
# falling output, rising unemployment and a larger, more negatively
# correlated covariance; the regime table reports the log-determinant
# of each regime's covariance and the largest root of its dynamics.

monthly = fetch(
    {"ip": "INDPRO", "u": "UNRATE", "nber": "USREC"},
    start="1959-10-01",
    end="2019-12-01",
    units={"ip": "cch", "u": "chg"},
)
names = ("ip", "du")
msvar = MarkovSwitchingVAR(
    monthly[["ip", "u"]].to_numpy(),
    order=2,
    n_regimes=2,
    switching_mean=True,
    switching_variance=True,
    names=names,
).fit(seed=1)
print(msvar.regime_table())
for index in range(msvar.n_regimes):
    regime = msvar.regime(index)
    print(f"regime {index}: intercepts {np.round(regime.deterministic[0], 3)}")
    print(f"           innovation covariance\n{np.round(regime.sigma_u, 4)}")

nber_monthly = monthly["nber"].to_numpy()[msvar.order :] > 0.5
probability = msvar.probabilities("smoothed")[:, 0]
qps = float(np.mean((probability - nber_monthly) ** 2))
concordance = float(np.mean((probability > 0.5) == nber_monthly))
print(f"MS-VAR     QPS {qps:.3f}   concordance {100 * concordance:5.1f}%")

# %%
# What to take away
# -----------------
# The regime-switching model is not told where the recessions are and
# finds most of them; where it disagrees with the NBER, the disagreement
# is informative about what the committee weighs beyond GDP. Two
# cautions carry over to any application. The labelling convention,
# ascending intercept, is what makes "regime 0" comparable across
# specifications; and the EM likelihood is multimodal, so the number of
# random starts is a parameter of the estimate, not a detail.
