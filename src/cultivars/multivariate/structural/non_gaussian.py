# filepath: /src/cultivars/multivariate/structural/non_gaussian.py
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

"""Non-Gaussian identification: independence does what zeros used to.

The Gaussian distribution is the only one a rotation cannot leave: rotate two
independent normal shocks and the result is two independent normal shocks,
which is precisely why every Gaussian scheme in this package needs an economic
restriction to choose among rotations. Drop normality and the symmetry
breaks -- with mutually independent shocks of which at most one is Gaussian,
the impact matrix is identified up to column order and sign from the data
alone (Comon 1994), and the SVAR literature has built estimators on exactly
this fact (Lanne, Meitz, and Saikkonen 2017; Gouriéroux, Monfort, and Renne
2017).

The estimator here is the moment route rather than a parametric likelihood:
whiten the innovations by the Cholesky factor, and find the rotation that
jointly diagonalizes the third- and fourth-order cumulant slices -- for
independent sources every one of them is diagonal in source coordinates, so
the search is the same joint-diagonalization surface the heteroskedasticity
scheme refines, warm-started at the eigenvectors of the kurtosis-weighted
covariance (the FOBI solution). Third-order slices carry skewness and
fourth-order slices carry tail weight, so a shock identified through either
is reachable.

As with heteroskedasticity, the data identify and the user labels: the
recovered shocks are statistical objects, ordered by how non-Gaussian they
are, and an economic name for any of them is a claim to be argued from
outside the model. The identification condition -- at most one Gaussian
shock -- is checked empirically and reported.

References:
    Comon, P. (1994). Independent component analysis, a new concept? *Signal
        Processing*, 36(3), 287-314.
    Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for non-Gaussian
        signals. *IEE Proceedings F*, 140(6), 362-370.
    Lanne, M., Meitz, M., & Saikkonen, P. (2017). Identification and
        estimation of non-Gaussian structural vector autoregressions.
        *Journal of Econometrics*, 196(2), 288-304.
    Gouriéroux, C., Monfort, A., & Renne, J.-P. (2017). Statistical inference
        for independent component analysis: Application to structural VAR
        models. *Journal of Econometrics*, 196(1), 111-126.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..._core import (
    ClosedSystemResult,
    _cumulant_slices,
    _lower_cholesky,
)
from ..._internals import _CoDiagonalObjective, _IdentificationModel, _solve
from ...exceptions import SpecificationError
from .zero_restrictions import SVARResult


class NonGaussianSVAR(_IdentificationModel[SVARResult]):
    """Identification from shock independence and non-Gaussianity, Comon (1994).

    The identifying assumptions are statistical and stated as such: the
    structural shocks are mutually *independent* -- strictly stronger than the
    uncorrelatedness every scheme imposes -- and at most one of them is
    Gaussian. Under them, the rotation of the whitened innovations that
    restores independence is unique up to column order and sign, and it is
    found by jointly diagonalizing the empirical third- and fourth-order
    cumulant slices.

    Shocks are unit variance, ordered by descending absolute excess kurtosis
    -- the most non-Gaussian first -- and labelled as the statistical objects
    they are. The summary reports each shock's skewness and excess kurtosis:
    they are the scheme's entire empirical content, and a shock with both
    inside sampling noise is one the data could not have separated, which the
    identification flag says plainly.

    Args:
        result: The fitted closed reduced-form result to identify.
        shock_names: One label per shock column, in descending-kurtosis order.
            Defaults to ``shock1 ... shockk``, deliberately not the variable
            names.

    Raises:
        SpecificationError: If the result is not a closed system, the labels
            are malformed, or the effective sample is too short for
            fourth-moment estimation.
        NumericalError: If the innovation covariance is not positive definite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> shocks = np.column_stack(
        ...     [rng.laplace(size=600), rng.uniform(-1.7, 1.7, size=600)]
        ... )
        >>> y = np.zeros((600, 2))
        >>> for t in range(1, 600):
        ...     y[t] = 0.4 * y[t - 1] + shocks[t]
        >>> svar = NonGaussianSVAR(VAR(y, order=1).fit()).identify()
        >>> svar.is_complete
        True
    """

    __slots__ = ("_labels",)

    def __init__(
        self,
        result: ClosedSystemResult,
        *,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system, the sample length, and the labels."""
        super().__init__(result)
        k = self.k_endog
        nobs_resid = int(np.asarray(result.resid).shape[0])
        if nobs_resid < 10 * k:
            raise SpecificationError(
                f"an effective sample of {nobs_resid} rows is too short to "
                f"estimate fourth-order cumulants of {k} shocks with any "
                f"reliability; this scheme needs at least {10 * k}."
            )
        if shock_names is None:
            self._labels = tuple(f"shock{j + 1}" for j in range(k))
        else:
            resolved = tuple(str(name) for name in shock_names)
            if len(resolved) != k or len(set(resolved)) != k:
                raise SpecificationError(f"shock_names must be {k} unique labels; got {resolved}.")
            self._labels = resolved

    def identify(self) -> SVARResult:
        """Recover the impact matrix by restoring shock independence.

        Returns:
            The complete structural result, shock columns in descending order
            of absolute excess kurtosis.

        Raises:
            NumericalError: If the innovation covariance is not positive
                definite.
        """
        k = self.k_endog
        resid = np.asarray(self.source.resid, dtype=np.float64)
        nobs = resid.shape[0]
        moment = resid.T @ resid / nobs
        factor = _lower_cholesky(moment, "the innovation second moment")
        whitened = np.linalg.solve(factor, resid.T).T

        kurtosis_weighted = (
            (whitened * np.sum(whitened**2, axis=1, keepdims=True)).T @ whitened / nobs
        )
        _, warm = np.linalg.eigh((kurtosis_weighted + kurtosis_weighted.T) / 2.0)

        slices = _cumulant_slices(whitened)
        objective = _CoDiagonalObjective(targets=tuple(warm.T @ target @ warm for target in slices))
        refinement, residual = _solve(objective)
        rotation = warm @ refinement

        shocks = whitened @ rotation
        kurtosis = (shocks**4).mean(axis=0) - 3.0
        order = np.argsort(np.abs(kurtosis))[::-1]
        rotation = rotation[:, order]
        kurtosis = kurtosis[order]

        impact = factor @ rotation
        for j in range(k):
            column = impact[:, j]
            if column[int(np.argmax(np.abs(column)))] < 0.0:
                impact[:, j] = -column
                rotation[:, j] = -rotation[:, j]
        skewness = ((whitened @ rotation) ** 3).mean(axis=0)

        skew_noise = 4.0 * float(np.sqrt(6.0 / nobs))
        kurt_noise = 4.0 * float(np.sqrt(24.0 / nobs))
        gaussian_count = int(
            np.sum((np.abs(skewness) < skew_noise) & (np.abs(kurtosis) < kurt_noise))
        )
        verdict = (
            "at most one Gaussian shock"
            if gaussian_count <= 1
            else f"WEAK: {gaussian_count} shocks statistically Gaussian"
        )
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=self._labels,
            scheme="non-Gaussian",
            restriction=(
                "The structural shocks are assumed mutually independent -- "
                "strictly stronger than uncorrelated -- with at most one of "
                "them Gaussian; those assumptions are the entire "
                "identification. Shocks are unit variance, ordered by "
                "descending absolute excess kurtosis, and labelled by their "
                "statistical behavior rather than by economics -- an economic "
                "name for any of them is a claim to be argued from outside "
                "the model. A shock whose skewness and excess kurtosis both "
                "sit inside sampling noise is one the data could not have "
                "separated."
            ),
            diagnostics=(
                ("Skewness", ", ".join(f"{value:.3f}" for value in skewness)),
                (
                    "Excess kurtosis",
                    ", ".join(f"{value:.3f}" for value in kurtosis),
                ),
                ("Co-diagonalization residual", f"{residual:.3e}"),
                ("Identification", verdict),
            ),
        )
