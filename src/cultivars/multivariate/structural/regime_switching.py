# filepath: /src/cultivars/multivariate/structural/regime_switching.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# [standard MIT license block]

"""Structural identification per regime: one restriction, M answers.

A Markov-switching VAR is ``M`` linear systems tied together by a chain, and
conditional on the regime each is a closed reduced form. Identification
therefore happens regime by regime: the same identifying restriction --
stated once -- is applied to every regime's own coefficients and covariance,
and what switches across regimes is the *answer*, not the assumption. That is
the Sims-Waggoner-Zha discipline: holding the restriction pattern fixed while
the parameters switch is what makes "the monetary shock in the volatile
regime" and "the monetary shock in the quiet regime" the same economic object
measured in two states of the world (Sims, Waggoner & Zha 2008).

This model deliberately does *not* subclass the family's identification base:
that base's contract is one closed system, and an MS-VAR has ``M`` of them.
Each regime view already satisfies the closed-system protocol on its own, so
any scheme applies to any single regime directly --
``RecursiveSVAR(msvar.regime(1))`` works verbatim -- and this class adds what
the one-regime route cannot: the guarantee that every regime was identified
by the *same* declaration, and a result that answers per regime under one
roof.

Two ways in, matching the family's composition pattern. By default
``identify`` runs the recursive scheme in every regime, under one ordering.
Alternatively, identify each regime view yourself with any point-identified
model -- the same scheme with different arguments, even different schemes if
you can defend that -- and pass the results as ``structurals``; the
declarations then live where you stated them, and this model only checks
provenance and packages.

One caveat is inherited from the reduced form and repeated here because it
now has structural weight: regime labels are a sorting convention over a
relabelling-invariant likelihood, so "regime 1's impact matrix" means
"the impact matrix of the regime the ordering convention calls 1".

References:
    Sims, C. A., Waggoner, D. F., & Zha, T. (2008). Methods for inference in
        large multiple-equation Markov-switching models. *Journal of
        Econometrics*, 146(2), 255-274.
    Lanne, M., Lutkepohl, H., & Maciejowska, K. (2010). Structural vector
        autoregressions with Markov switching. *Journal of Economic Dynamics
        and Control*, 34(2), 121-131.
"""
