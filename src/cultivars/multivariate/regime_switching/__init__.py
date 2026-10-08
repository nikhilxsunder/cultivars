"""Latent-regime multivariate models: the regime inferred, never observed.

Every model here lets some block of a linear system switch with a
first-order Markov chain that is not in the data. Where
:mod:`~cultivars.multivariate.nonlinear` computes its regime from an
observable -- a threshold crossed, a transition function evaluated -- the
chain here is a latent state, and what estimation returns is a posterior
over regimes at every date: filtered for the real-time view, smoothed for
the chronology after the fact. Every quantity that depends on the regime,
from fitted values to the regime a date "is in", is an expectation under
that posterior rather than a fact about the data, and the results say so.

One module, :mod:`~cultivars.multivariate.regime_switching.markov_switching`,
holds three models. ``MarkovSwitchingVAR`` lets the intercepts, the lag
coefficients and the innovation covariance of a VAR switch in any
combination, estimated by EM on the Hamilton filter and Kim smoother with
multi-start screening, and exposes each regime as a closed linear system.
``MarkovSwitchingSVAR`` applies one identifying declaration to every one of
those systems and packages the per-regime structural answers.
``MarkovSwitchingDFM`` is the Kim-Nelson coincident index: one factor whose
intercept switches, estimated through the continuous-state Kim filter, with
the smoothed low-state probability as the business-cycle chronology. The
scalar family on the same filter is
:mod:`~cultivars.univariate.regime_switching`.

Two properties of mixtures are enforced rather than explained away. The
likelihood is invariant to relabelling the regimes, so every fit imposes a
sorting convention and records it. And the number of regimes is not testable
by a likelihood ratio -- under the null of fewer regimes the extra regime's
parameters are unidentified and its transition probabilities sit on the
boundary -- so that comparison is refused while tests holding the regime
count fixed remain available.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.regime_switching import markov_switching
    >>> rng = np.random.default_rng(0)
    >>> s = np.repeat([0, 1, 0, 1], 100)
    >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
    >>> res = markov_switching.MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
    >>> res.specification, bool((res.most_likely_regime == s[1:]).mean() > 0.98)
    ('MSIH(2)-VAR(1)', True)
"""

from . import markov_switching

__all__ = [
    "markov_switching",
]
