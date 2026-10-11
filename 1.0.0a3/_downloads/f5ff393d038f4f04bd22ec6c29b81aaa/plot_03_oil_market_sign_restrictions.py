# filepath: /examples/plot_03_oil_market_sign_restrictions.py
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
Oil supply and demand shocks: sign, narrative and set identification
====================================================================

Kilian (2009) decomposes oil price movements into three shocks -- oil
supply, aggregate demand, and oil-specific demand -- in a monthly VAR
of oil production, real activity and the real oil price. Recursive
identification works there only because of a strong assumption about
supply's within-month inelasticity; sign restrictions (Kilian and
Murphy 2012) replace it with the qualitative content of the economics:
a negative supply shock lowers production and activity and raises the
price; an aggregate demand shock raises all three; an oil-specific
demand shock raises production and price but lowers activity.

This example runs that scheme three ways and compares what each
delivers. Plain sign restrictions give an identified *set* summarized by
quantiles of accepted draws. Narrative restrictions (Antolin-Diaz and
Rubio-Ramirez 2018) add the historical fact that the August 1990
invasion of Kuwait was a negative supply shock, and shrink the set. The
exact bounds of Giacomini and Kitagawa (2021) show, for the supply
shock alone, how wide the set really is, independently of the prior the
random rotations impose on it.

The data are U.S. series only: U.S. field production of crude oil
stands in for world production, and industrial production for the
global activity index. Both substitutions matter and the closing
section says how.
"""

# %%
# Data
# ----
# Monthly, 1973 to 2019: U.S. crude production in thousand barrels per
# day, industrial production, the WTI spot price and the CPI. Production
# enters as the percent change, activity as 100 times its log, and the
# price as 100 times the log of WTI deflated by the CPI.

import matplotlib.pyplot as plt
import numpy as np

from cultivars.exceptions import SpecificationError
from cultivars.multivariate.reduced_form.vector_autoregression import VAR
from cultivars.multivariate.structural.set_identification import SetIdentifiedSVAR
from cultivars.multivariate.structural.sign_restrictions import (
    NarrativeSignRestrictedSVAR,
    SignRestrictedSVAR,
)

from _fred import fetch

raw = fetch(
    {"prod": "IPG211111CS", "ip": "INDPRO", "wti": "WTISPLC", "cpi": "CPIAUCSL"},
    start="1972-12-01",
    end="2019-12-01",
)
data = raw[["prod"]].copy()
data["prod"] = 100.0 * np.log(raw["prod"]).diff()
data["act"] = 100.0 * np.log(raw["ip"])
data["rpo"] = 100.0 * np.log(raw["wti"] / raw["cpi"])
data = data.dropna()
names = ("prod", "act", "rpo")
print(data.describe().round(2))

# %%
# Reduced form
# ------------
# Kilian uses 24 lags to let the slow adjustment of production through.
# Twelve is enough for the point of this example and keeps the rotation
# search fast; the portmanteau test says whether it is too few.

order = 12
var = VAR(data.to_numpy(), order=order, names=names).fit()
print(var.residual_diagnostics(portmanteau_lags=24, arch_lags=12))

# %%
# Sign restrictions
# -----------------
# The restriction table names the three shocks and the sign of each
# variable's impact response. The signs are imposed at horizons 0 and 1,
# so a draw must satisfy them for two months. The identified set is
# summarized by the 16th, 50th and 84th percentiles of the accepted draws
# at each horizon.

signs = {
    "supply": {"prod": "-", "act": "-", "rpo": "+"},
    "agg_demand": {"prod": "+", "act": "+", "rpo": "+"},
    "oil_demand": {"prod": "+", "act": "-", "rpo": "+"},
}
sign_set = SignRestrictedSVAR(var, signs, horizons=(0,), draws=500, seed=3).identify()
print(sign_set.summary())
print(f"Acceptance rate {100 * sign_set.acceptance_rate:.2f}% over {sign_set.attempts} rotations")

horizon = 24
bands = sign_set.irf(horizon, quantiles=(0.16, 0.5, 0.84), cumulative=False)

# %%
# Narrative restriction
# ---------------------
# The invasion of Kuwait on 2 August 1990 took Iraqi and Kuwaiti output
# off the market. Antolin-Diaz and Rubio-Ramirez impose two facts about
# that month: the supply shock was negative -- which in the convention
# above means a *positive* value of the shock labelled ``supply``, since
# that shock is defined as a production cut -- and it was the most
# important contributor to the unexpected movement in production. Events
# are indexed on the effective sample, after the lags.
#
# Narrative events can contradict the data at a given reduced form, in
# which case no rotation survives and the model refuses rather than
# returning an empty set. The attempt budget bounds the search; the
# fallback below keeps the example running so the comparison can still
# be drawn from the plain sign-restricted set.

dates = data.index[order:]
event = int(np.flatnonzero(dates == "1990-08-01")[0])
try:
    narrative = NarrativeSignRestrictedSVAR(
        var,
        signs,
        shock_signs=[("supply", event, "+")],
        contributions=[("supply", "prod", event, "most")],
        horizons=(0,),
        draws=300,
        max_attempts=150_000,
        seed=3,
    ).identify()
except SpecificationError as refusal:
    print(f"Narrative identification refused: {refusal}")
    narrative_label = "narrative (refused; signs only)"
    narrative_bands = bands
else:
    print(narrative.summary())
    print(
        f"Acceptance rate {100 * narrative.acceptance_rate:.2f}% "
        f"over {narrative.attempts} rotations"
    )
    narrative_label = "narrative median"
    narrative_bands = narrative.irf(horizon, quantiles=(0.16, 0.5, 0.84))

# %%
# Exact bounds for the supply shock
# ---------------------------------
# Giacomini and Kitagawa's bounds are the extreme responses over *every*
# rotation satisfying the supply-shock signs, computed exactly rather
# than by sampling. They are what the restrictions alone imply; the
# distance between them and the sign-restricted quantile bands is the
# contribution of the uniform prior over rotations.

bounds = SetIdentifiedSVAR(var, signs["supply"], shock="supply", horizons=(0,)).identify()
print(bounds.summary())
exact = bounds.irf(horizon)

# %%
# Responses to a negative oil supply shock
# ----------------------------------------
# Three readings of the same shock: the quantile band of the
# sign-restricted set, the narrower band once the Kuwait event is
# imposed, and the exact envelope of everything the signs allow.

labels = {
    "prod": "Oil production (%)",
    "act": "Industrial production (%)",
    "rpo": "Real oil price (%)",
}
steps = np.arange(horizon + 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
supply = list(sign_set.shock_names).index("supply")
for i, (ax, name) in enumerate(zip(axes, names, strict=True)):
    lower, upper = bands[0, :, i, supply], bands[2, :, i, supply]
    ax.fill_between(steps, exact[:, i, 0], exact[:, i, 1], color="grey", alpha=0.2, label="exact")
    ax.fill_between(steps, lower, upper, color="C0", alpha=0.3, label="signs 68%")
    ax.plot(steps, bands[1, :, i, supply], color="C0", lw=2, label="signs median")
    ax.plot(steps, narrative_bands[1, :, i, supply], color="C3", lw=2, label=narrative_label)
    ax.plot(steps, narrative_bands[0, :, i, supply], color="C3", lw=1, ls="--")
    ax.plot(steps, narrative_bands[2, :, i, supply], color="C3", lw=1, ls="--")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title(labels[name])
    ax.set_xlabel("months after shock")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Negative oil supply shock under three identification schemes")
fig.tight_layout()

# %%
# Variance shares of the real price
# ---------------------------------
# Kilian's finding is that oil-specific demand shocks, not supply, drive
# most of the real price variance at business-cycle horizons. The median
# shares below put that to the sign-identified set. Medians are taken
# draw by draw per shock, so the three need not sum to one.

fevd = sign_set.fevd(horizon)
rpo = names.index("rpo")
print("Median share of the real oil price forecast error variance")
for h in (1, 6, 12, 24):
    shares = ", ".join(
        f"{shock}: {100 * fevd[1, h, rpo, j]:5.1f}%" for j, shock in enumerate(sign_set.shock_names)
    )
    print(f"  h = {h:2d}   {shares}")

# %%
# What to take away
# -----------------
# The sign-restricted band is not a confidence band: its width mixes
# sampling uncertainty about the reduced form, which it ignores, with
# identification uncertainty, which it summarizes under one particular
# prior. The exact bounds strip the prior out and are usually
# uncomfortably wide. Narrative restrictions are the cheapest way to
# narrow them, and the acceptance rates printed above say how much of
# the identified set a single historical fact removes.
#
# The two data substitutions are not innocent. U.S. production is a
# small and, after 2010, structurally different share of world supply,
# and industrial production is a poor proxy for the global activity
# index Kilian builds from shipping freight rates. Replicating the paper
# properly needs the global series, which are not on FRED; the point of
# this example is the identification workflow, not the point estimates.
