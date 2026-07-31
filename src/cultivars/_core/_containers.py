

from typing import NamedTuple

import numpy as np

class InformationCriteria(NamedTuple):
    """Model-selection criteria.

    Attributes:
        aic: Akaike information criterion.
        bic: Bayesian (Schwarz) information criterion.
        hqic: Hannan-Quinn information criterion.
    """

    aic: float
    bic: float
    hqic: float


def information_criteria(llf: float, nobs: int, n_params: int) -> InformationCriteria:
    """Compute AIC/BIC/HQIC from a log-likelihood and parameter count."""
    aic = -2.0 * llf + 2.0 * n_params
    bic = -2.0 * llf + n_params * np.log(nobs)
    hqic = -2.0 * llf + 2.0 * n_params * np.log(np.log(nobs))
    return InformationCriteria(float(aic), float(bic), float(hqic))