# filepath: /src/cultivars/engine/_core/_notes.py
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

from __future__ import annotations

__all__ = [
    "_AGGREGATION_NOTE",
    "_CHOLESKY_NOTE",
    "_CONDITIONAL_REFUSAL",
    "_HR_CONDITIONAL_NOTE",
    "_LABEL_NOTE",
    "_MIDAS_CONDITIONAL_NOTE",
    "_NARRATIVE_NOTE",
    "_NO_CLOSED_SYSTEM",
    "_NULL",
    "_PARTIAL_IDENTIFICATION_NOTE",
    "_SCALE_NOTE",
    "_SET_BOUNDS_NOTE",
    "_SIGN_QUANTILE_NOTE",
    "_UNIT_SHOCK_NOTE",
    "_UNSTABLE_NOTE",
    "_VARMA_IDENTIFICATION_NOTE",
]

_CHOLESKY_NOTE: str = (
    "Orthogonalized impulse responses and the variance decomposition use a Cholesky "
    "factor, which imposes the recursive ordering of `names`; that is a structural "
    "assumption, not a reduced-form result."
)

_UNSTABLE_NOTE: str = "NOT STABLE: impulse responses diverge and forecasts are meaningless."


_CONDITIONAL_REFUSAL: str = (
    "a conditional vector error-correction model has no closed system, so {what} is not "
    "defined for it. The weakly exogenous block {names} is carried without equations, "
    "which means there is no law of motion to propagate a shock through and no companion "
    "matrix to take roots of. Closing the system is what a global vector autoregression "
    "does, by stacking units and solving the links; forecast(), which only needs a path "
    "for x rather than a model of it, is available here."
)

_NO_CLOSED_SYSTEM: str = (
    "{model} is a conditional model: {what} needs a law of motion for every variable in "
    "the system, and this specification deliberately provides none for its exogenous "
    "block. That omission is the model, not a gap in it. Everything estimated -- "
    "coefficients, standard errors, p-values, residual diagnostics -- is available; "
    "closing the system is what a global vector autoregression does by linking units."
)

_AGGREGATION_NOTE: str = (
    "One or more series is temporally aggregated. Coefficients and the "
    "innovation covariance are taken as given: see MFVAR for why they are not "
    "estimated from the mixed-frequency sample."
)

_MIDAS_CONDITIONAL_NOTE: str = (
    "Coefficient standard errors condition on the estimated lag polynomial. "
    "Call joint_stderr() for errors that also account for estimating it."
)

_HR_CONDITIONAL_NOTE: str = (
    "Standard errors treat the lagged innovations in the design as observed "
    "regressors rather than as the estimates they are; they are conditional in "
    "the Hannan-Rissanen sense and modestly understate uncertainty."
)

_VARMA_IDENTIFICATION_NOTE: str = (
    "An unrestricted VARMA(p, q) is not globally identified: distinct (A, M) "
    "pairs can generate identical second moments, and echelon-form restrictions "
    "-- the standard resolution -- are not imposed here. For a stable, "
    "invertible representation the moving-average matrices, forecasts, impulse "
    "responses, and variance decompositions are invariant across "
    "observationally equivalent parameterizations; the individual coefficients "
    "are not."
)

_PARTIAL_IDENTIFICATION_NOTE: str = (
    "Only the listed shock columns are identified; the remaining structural "
    "shocks exist but are not pinned down by these restrictions. Variance "
    "shares therefore need not sum to one across the identified shocks, and "
    "the unidentified remainder is exactly the variation the scheme is silent "
    "about."
)

_SIGN_QUANTILE_NOTE: str = (
    "Bands are pointwise quantiles across the accepted rotations. No single "
    "structural model traces the median band: at each horizon the quantile may "
    "come from a different rotation, which is the Fry-Pagan critique, and the "
    "honest reading is as a summary of the identified set rather than as the "
    "impulse response of a representative model."
)

_UNIT_SHOCK_NOTE: str = (
    "Shocks are normalized to unit variance, so impact-column entries are "
    "responses to a one-standard-deviation structural shock."
)

_NARRATIVE_NOTE: str = (
    "Narrative events were checked against the point-estimate residuals: the "
    "set is the rotations consistent with the declared signs and the declared "
    "history at this reduced form. The Antolin-Diaz and Rubio-Ramirez "
    "importance weighting, which propagates narrative information into the "
    "reduced-form posterior, belongs to the posterior-draw version that "
    "arrives with the sampling backend."
)

_SET_BOUNDS_NOTE: str = (
    "Bounds are the exact endpoints of the identified set, computed pointwise "
    "per response and horizon: they depend on no prior over rotations, which "
    "is what separates them from the quantile bands of a sign-restricted set "
    "-- those inherit the uniform prior over the admissible rotations, the "
    "Baumeister-Hamilton critique. Read the bounds pointwise: the set of "
    "admissible impulse-response paths is not the box they outline. Computed "
    "at the reduced-form point estimate; the robust Bayesian treatment, "
    "which repeats these bounds across posterior draws, arrives with the "
    "sampling backend."
)

_SCALE_NOTE: str = (
    "Each impact column is the response to its shock at the shock's in-sample "
    "average variance: log-variance paths have zero sample mean and their "
    "level sits in B. Shocks are not unit-variance; volatility_path() has "
    "the time variation."
)

_LABEL_NOTE: str = (
    "Shock j is the one loading most on variable j, signed so the diagonal "
    "is positive: a convention, not an economic identification. An economic "
    "name for any shock is a claim to be argued from outside the model."
)

_NULL: str = "linear AR({order}) is correctly specified"
