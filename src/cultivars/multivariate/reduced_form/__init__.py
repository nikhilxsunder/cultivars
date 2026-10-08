"""Reduced-form multivariate models: one least-squares surface, eight designs.

Every model here writes a vector of observables as a linear function of its
own past plus an innovation whose covariance is unrestricted, estimates that
function without imposing a structural interpretation on the innovations,
and hands the result to a shared post-estimation surface -- forecasts,
impulse responses, variance and historical decompositions, Granger
causality, stability, residual diagnostics, coefficient inference,
information criteria and model comparison. The modules differ in what sits
on the right-hand side and in how the system is solved, not in what the
fitted object can then be asked.

The root is :mod:`~cultivars.multivariate.reduced_form.vector_autoregression`
(``VAR``, ``VARX``): least squares on a lag design, with an optional
distributed lag of weakly exogenous regressors. Four modules keep that
design and change the solve or the restriction.
:mod:`~cultivars.multivariate.reduced_form.error_correction` (``VECM``,
``VECMX``) imposes a reduced-rank long-run matrix for cointegrated levels
and returns to the levels VAR on request;
:mod:`~cultivars.multivariate.reduced_form.moving_average` (``VARMA``) adds
a moving-average block and estimates it by Hannan-Rissanen regressions
rather than a likelihood search;
:mod:`~cultivars.multivariate.reduced_form.panel` (``PanelVAR``) pools the
slopes across units behind unit-specific intercepts; and
:mod:`~cultivars.multivariate.reduced_form.closed_global` (``GVAR``) does
not estimate at all -- it takes fitted conditional units and a weight
matrix and closes them into one system, so its result carries the
propagation surface and the weak-exogeneity diagnostics but no
coefficient inference. Three modules change the observation side and
estimate through a state space instead of a regression:
:mod:`~cultivars.multivariate.reduced_form.mixed_frequency` (``MFVAR``,
``MIDASVAR``) reads a panel sampled at more than one frequency,
:mod:`~cultivars.multivariate.reduced_form.functional` (``FunctionalVAR``)
reads curves through a basis, and
:mod:`~cultivars.multivariate.reduced_form.term_structure`
(``DynamicNelsonSiegel``, ``TimeVaryingNelsonSiegel``) reads yield curves
through the Nelson-Siegel loadings.

The structural step -- declaring what the innovations mean -- is
deliberately absent from every result here. Each summary says that
orthogonalized responses rest on the recursive ordering of ``names`` and
points at :mod:`~cultivars.multivariate.structural`, where an
identification is declared rather than implied.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form import vector_autoregression, panel
    >>> rng = np.random.default_rng(0)
    >>> y = rng.standard_normal((120, 2))
    >>> single = vector_autoregression.VAR(y, order=1).fit()
    >>> pooled = panel.PanelVAR([y[:60], y[60:]], order=1).fit()
    >>> single.irf(2).shape == pooled.irf(2).shape, single.k_endog, pooled.n_units
    (True, 2, 2)
"""

from . import (
    closed_global,
    error_correction,
    functional,
    mixed_frequency,
    moving_average,
    panel,
    term_structure,
    vector_autoregression,
)

__all__ = [
    "closed_global",
    "error_correction",
    "functional",
    "mixed_frequency",
    "moving_average",
    "panel",
    "term_structure",
    "vector_autoregression",
]
