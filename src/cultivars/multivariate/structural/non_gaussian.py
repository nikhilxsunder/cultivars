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
r"""Non-Gaussian identification: independence does what zeros used to.

Every impact matrix consistent with the innovation covariance is
:math:`B = L Q` for the Cholesky factor :math:`L` and some orthogonal
:math:`Q`, and the Gaussian distribution is the only one a rotation cannot
leave: rotate two independent normal shocks and the result is two
independent normal shocks, which is precisely why every Gaussian scheme in
this package needs an economic restriction to choose :math:`Q`. Drop
normality and the symmetry breaks. With mutually independent shocks of which
at most one is Gaussian, :math:`Q` is identified up to column order and sign
from the data alone (Comon 1994), and the SVAR literature has built
estimators on exactly this fact (Lanne, Meitz and Saikkonen 2017;
Gouriéroux, Monfort and Renne 2017). The estimator here is the moment route
rather than a parametric likelihood: whiten the innovations, and find the
rotation that jointly diagonalizes the third- and fourth-order cumulant
slices,

.. math::

   C^{(3)}_{\cdot\cdot j} = \mathbb{E}[w_i w_k w_j], \qquad
   C^{(4)}_{\cdot\cdot jl} = \mathbb{E}[w_i w_k w_j w_l]
   - \delta_{ik}\delta_{jl} - \delta_{ij}\delta_{kl} - \delta_{il}\delta_{jk},

every one of which is diagonal in source coordinates when the sources are
independent. Third-order slices carry skewness and fourth-order slices carry
tail weight, so a shock identified through either is reachable, and the
search is warm-started at the eigenvectors of the kurtosis-weighted
covariance -- the FOBI solution -- on the same joint-diagonalization surface
the heteroskedasticity scheme refines.

Two commitments shape the surface. First, the data identify and the user
labels. The recovered shocks are statistical objects, ordered by how
non-Gaussian they are and named ``shock1 ... shockk`` unless the user says
otherwise; an economic name for any of them is a claim to be argued from
outside the model, and the restriction note on every summary says so.
Second, the identification condition is checked, not assumed. Each shock's
skewness and excess kurtosis are reported against their sampling noise, a
count of statistically Gaussian shocks above one is flagged ``WEAK``, and the
residual off-diagonal energy after joint diagonalization is reported as the
data's verdict on independence itself -- the assumption that is strictly
stronger than uncorrelatedness, and the one common volatility across shocks
would violate.

Layout. :class:`NonGaussianSVAR` is an ``_IdentificationModel`` from
``_internals``; it whitens through ``_lower_cholesky`` in ``_core``, builds
the slices with ``_cumulant_slices`` in ``_core``, and minimizes the shared
``_CoDiagonalObjective`` from ``_internals`` with ``_solve``, then packages
the rotation into ``SVARResult`` from
:mod:`~cultivars.multivariate.structural.zero_restrictions` with
``scheme="non-Gaussian"``. The second-moment route to the same kind of
statistical identification is
:mod:`~cultivars.multivariate.structural.heteroskedacity`.

References:
    Comon, P. (1994). Independent component analysis, a new concept? *Signal
    Processing*, 36(3), 287-314.

    Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for
    non-Gaussian signals. *IEE Proceedings F*, 140(6), 362-370.

    Lanne, M., Meitz, M., & Saikkonen, P. (2017). Identification and
    estimation of non-Gaussian structural vector autoregressions. *Journal of
    Econometrics*, 196(2), 288-304.

    Gouriéroux, C., Monfort, A., & Renne, J.-P. (2017). Statistical inference
    for independent component analysis: Application to structural VAR
    models. *Journal of Econometrics*, 196(1), 111-126.

Example:
    The same bivariate system identified twice, once with Laplace and
    uniform shocks and once with Gaussian ones. The first recovers the impact
    columns from independence alone; the second returns a rotation the data
    could not have chosen, and says so:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
    >>> def simulate(eps):
    ...     y = np.zeros((602, 2))
    ...     for t in range(1, 602):
    ...         y[t] = 0.4 * y[t - 1] + B @ eps[t]
    ...     return VAR(y, order=1).fit()
    >>> heavy = rng.laplace(size=602) / np.sqrt(2)
    >>> flat = rng.uniform(-np.sqrt(3), np.sqrt(3), size=602)
    >>> sharp = NonGaussianSVAR(simulate(np.column_stack([heavy, flat]))).identify()
    >>> blunt = NonGaussianSVAR(simulate(rng.standard_normal((602, 2)))).identify()
    >>> dict(sharp.diagnostics)["Identification"], dict(blunt.diagnostics)["Identification"]
    ('at most one Gaussian shock', 'WEAK: 2 shocks statistically Gaussian')
    >>> unit = lambda v: v / np.linalg.norm(v)
    >>> bool(np.abs(unit(sharp.impact[:, 0]) - unit(B[:, 0])).max() < 0.07)
    True
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

__all__ = ["NonGaussianSVAR"]


class NonGaussianSVAR(_IdentificationModel[SVARResult]):
    r"""Identification from shock independence and non-Gaussianity, Comon (1994).

    Whiten the innovations by the Cholesky factor of their second moment,
    :math:`w_t = L^{-1} u_t`, so that any impact matrix consistent with the
    covariance is :math:`B = L Q` for an orthogonal :math:`Q`. If the
    structural shocks :math:`\varepsilon_t = Q' w_t` are mutually
    *independent* and at most one is Gaussian, :math:`Q` is unique up to
    column order and sign: independence makes every third- and fourth-order
    cumulant slice of the shocks diagonal, and only the Gaussian direction
    leaves those slices invariant under rotation. The estimator finds the
    :math:`Q` that jointly diagonalizes the empirical cumulant slices of
    :math:`w_t`, warm-started at the eigenvectors of the kurtosis-weighted
    covariance (the FOBI solution).

    Shocks are unit variance, ordered by descending absolute excess kurtosis
    -- the most non-Gaussian first -- and labelled as the statistical objects
    they are. The summary reports each shock's skewness and excess kurtosis:
    they are the scheme's entire empirical content, and a shock with both
    inside sampling noise is one the data could not have separated, which the
    identification flag says plainly.

    Note:
        The identifying assumptions are statistical and stated as such.
        Independence is strictly stronger than the uncorrelatedness every
        scheme imposes -- it rules out, for instance, common volatility
        across shocks, which makes the shocks dependent through their second
        moments even when uncorrelated -- and the at-most-one-Gaussian
        condition is what the ``Identification`` diagnostic checks. As with
        heteroskedasticity, the data identify and the user labels: the
        default names are ``shock1 ... shockk`` on purpose, and an economic
        name for a column is a claim argued from outside the model.

    Attributes:
        _source: The closed reduced-form result being identified.
        _labels: The shock labels, in descending-kurtosis column order.

    See Also:
        * :class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
          -- the complete result returned.
        * :class:`~cultivars.multivariate.structural.heteroskedacity.HeteroskedasticSVAR`
          -- identification from second-moment shifts across declared
          regimes, on the same joint-diagonalization surface.

    References:
        Comon, P. (1994). Independent component analysis, a new concept?
        *Signal Processing*, 36(3), 287-314.

        Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for
        non-Gaussian signals. *IEE Proceedings F*, 140(6), 362-370.

        Lanne, M., Meitz, M., & Saikkonen, P. (2017). Identification and
        estimation of non-Gaussian structural vector autoregressions.
        *Journal of Econometrics*, 196(2), 288-304.

        Gouriéroux, C., Monfort, A., & Renne, J.-P. (2017). Statistical
        inference for independent component analysis: Application to
        structural VAR models. *Journal of Econometrics*, 196(1), 111-126.

    Example:
        A heavy-tailed shock and a flat-tailed one mixed through a non-
        triangular impact matrix. Independence recovers both columns up to
        scale with no economic restriction, the heavy-tailed shock comes
        first, and the recovered shock series track the truth:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
        >>> heavy = rng.laplace(size=602) / np.sqrt(2)
        >>> flat = rng.uniform(-np.sqrt(3), np.sqrt(3), size=602)
        >>> eps = np.column_stack([heavy, flat])
        >>> y = np.zeros((602, 2))
        >>> for t in range(1, 602):
        ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("a", "b")).fit()
        >>> svar = NonGaussianSVAR(res, shock_names=("heavy", "flat")).identify()
        >>> svar.is_complete, svar.shock_names, dict(svar.diagnostics)["Identification"]
        (True, ('heavy', 'flat'), 'at most one Gaussian shock')
        >>> dict(svar.diagnostics)["Excess kurtosis"]
        '3.249, -1.196'
        >>> unit = lambda v: v / np.linalg.norm(v)
        >>> bool(np.abs(unit(svar.impact[:, 0]) - unit(B[:, 0])).max() < 0.07)
        True
        >>> recovered = svar.structural_shocks()
        >>> bool(np.corrcoef(recovered[:, 1], flat[1:])[0, 1] > 0.99)
        True
    """

    __slots__ = ("_labels",)

    def __init__(
        self,
        result: ClosedSystemResult,
        *,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system, the sample length, and the labels.

        Args:
            result: The fitted closed reduced-form result to identify.
            shock_names: One label per shock column, in descending-kurtosis
                order. Defaults to ``shock1 ... shockk``, deliberately not the
                variable names.

        Raises:
            SpecificationError: If the result is not a closed system, the
                labels are malformed, or the effective sample has fewer than
                ``10 k`` rows -- too few for fourth-order cumulants.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1).fit()
            >>> NonGaussianSVAR(res, shock_names=("u", "v"))._labels
            ('u', 'v')
            >>> NonGaussianSVAR(res, shock_names=("u",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: shock_names must be 2 unique labels; ...
            >>> short = VAR(rng.standard_normal((15, 2)), order=1).fit()
            >>> NonGaussianSVAR(short)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: an effective sample of 14 rows is too short ...
        """
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

        Whitens the residuals, warm-starts at the FOBI eigenvectors, refines
        them by joint diagonalization of the third- and fourth-order
        cumulant slices, orders the columns by descending absolute excess
        kurtosis, and signs each column so its largest entry is positive.

        Returns:
            The complete structural result, shock columns in descending order
            of absolute excess kurtosis, with each shock's skewness and
            excess kurtosis, the co-diagonalization residual and the
            Gaussianity verdict in its diagnostics.

        Raises:
            NumericalError: If the innovation covariance is not positive
                definite.

        Example:
            Gaussian innovations carry no identifying information: the
            scheme still returns a rotation, and the verdict says the data
            could not have chosen it:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> y = np.zeros((602, 2))
            >>> for t in range(1, 602):
            ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1).fit()
            >>> svar = NonGaussianSVAR(res).identify()
            >>> svar.is_complete, dict(svar.diagnostics)["Identification"]
            (True, 'WEAK: 2 shocks statistically Gaussian')
            >>> bool(np.allclose(svar.impact @ svar.impact.T, res.resid.T @ res.resid / 601))
            True
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
