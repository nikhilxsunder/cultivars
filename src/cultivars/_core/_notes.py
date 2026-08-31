_CHOLESKY_NOTE = (
    "Orthogonalized impulse responses and the variance decomposition use a Cholesky "
    "factor, which imposes the recursive ordering of `names`; that is a structural "
    "assumption, not a reduced-form result."
)

_UNSTABLE_NOTE = "NOT STABLE: impulse responses diverge and forecasts are meaningless."


_CONDITIONAL_REFUSAL = (
    "a conditional vector error-correction model has no closed system, so {what} is not "
    "defined for it. The weakly exogenous block {names} is carried without equations, "
    "which means there is no law of motion to propagate a shock through and no companion "
    "matrix to take roots of. Closing the system is what a global vector autoregression "
    "does, by stacking units and solving the links; forecast(), which only needs a path "
    "for x rather than a model of it, is available here."
)

_NO_CLOSED_SYSTEM = (
    "{model} is a conditional model: {what} needs a law of motion for every variable in "
    "the system, and this specification deliberately provides none for its exogenous "
    "block. That omission is the model, not a gap in it. Everything estimated -- "
    "coefficients, standard errors, p-values, residual diagnostics -- is available; "
    "closing the system is what a global vector autoregression does by linking units."
)

_AGGREGATION_NOTE = (
    "One or more series is temporally aggregated. Coefficients and the "
    "innovation covariance are taken as given: see MFVAR for why they are not "
    "estimated from the mixed-frequency sample."
)

_MIDAS_CONDITIONAL_NOTE = (
    "Coefficient standard errors condition on the estimated lag polynomial. "
    "Call joint_stderr() for errors that also account for estimating it."
)

_HR_CONDITIONAL_NOTE = (
    "Standard errors treat the lagged innovations in the design as observed "
    "regressors rather than as the estimates they are; they are conditional in "
    "the Hannan-Rissanen sense and modestly understate uncertainty."
)

_VARMA_IDENTIFICATION_NOTE = (
    "An unrestricted VARMA(p, q) is not globally identified: distinct (A, M) "
    "pairs can generate identical second moments, and echelon-form restrictions "
    "-- the standard resolution -- are not imposed here. For a stable, "
    "invertible representation the moving-average matrices, forecasts, impulse "
    "responses, and variance decompositions are invariant across "
    "observationally equivalent parameterizations; the individual coefficients "
    "are not."
)
