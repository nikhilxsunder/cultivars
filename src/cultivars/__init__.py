"""Cultivars: research-grade time series modelling.

The public API is organised by subpackage; import the namespace you need::

    import cultivars as cv
    model = cv.univariate.ARMA(...)
    from cultivars.multivariate.reduced_form import VAR

Nothing below the subpackage level is re-exported here.
"""

from __future__ import annotations

from . import (
    bayes,
    diagnostics,
    forecast,
    multivariate,
    spectral,
    state_space,
    typing,
    univariate,
)

__all__ = [
    "bayes",
    "diagnostics",
    "forecast",
    "multivariate",
    "spectral",
    "state_space",
    "typing",
    "univariate",
]
