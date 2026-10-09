# filepath: /src/cultivars/multivariate/structural/heteroskedasticity.py
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
r"""Identification through heteroskedasticity: the data identify, the user labels.

Every other scheme in this package buys identification with an economic
restriction someone must defend. This one buys it with a statistical fact.
If the structural shocks' variances shift across declared regimes while the
impact matrix stays constant,

.. math::

   \Sigma_m = B \Lambda_m B', \qquad m = 1, \dots, M,

with :math:`\Lambda_m` diagonal, then the regime covariances jointly pin
down :math:`B` up to column order and sign, and no zero, no sign pattern, no
instrument is needed (Rigobon 2003). Normalizing :math:`\Lambda_1 = I` and
writing :math:`\Sigma_1 = L L'`, the whitened second-regime covariance
:math:`L^{-1} \Sigma_2 L^{-\top} = Q \Lambda_2 Q'` is a symmetric
eigenproblem whose eigenvectors are the rotation and whose eigenvalues are
the variance ratios, and :math:`B = L Q`. With two regimes that is the whole
estimator. With more, :math:`B` is over-identified: no single rotation
exactly diagonalizes every regime in sample, so the estimate comes from
joint approximate diagonalization warm-started at the two-regime solution,
and the residual off-diagonal energy is reported as the data's verdict on
the constant-impact assumption.

Two commitments shape the surface. First, what the data cannot supply is
not invented. The recovered shocks are labelled by their variance behaviour
-- unit variance in the first regime, ordered by descending ratio in the
second, named ``shock1 ... shockk`` unless the user says otherwise -- and
attaching a name like "monetary" to one of them is a claim the user must
argue from outside the model, which the restriction note on every summary
says plainly. Second, the identification condition is checked, not assumed.
Identification needs the variance ratios to be distinct: two shocks whose
variances shift by the same factor are indistinguishable within their span,
so the minimum ratio separation is reported, a separation within sampling
noise is flagged ``WEAK``, and an exact tie is a refusal rather than an
arbitrary rotation.

Layout. :class:`HeteroskedasticSVAR` is an ``_IdentificationModel`` from
``_internals``; the regime assignment is validated by ``_validate_regimes``
in ``_core``, the first regime's factor comes from ``_lower_cholesky`` in
``_core``, the multi-regime refinement minimizes the shared
``_CoDiagonalObjective`` from ``_internals`` with ``_solve``, and the
rotation is packaged into ``SVARResult`` from
:mod:`~cultivars.multivariate.structural.zero_restrictions` with
``scheme="heteroskedasticity"``. The same idea with the variance path
estimated rather than declared is
:mod:`~cultivars.multivariate.structural.stochastic_volatility`;
identification from higher moments on the same diagonalization surface is
:mod:`~cultivars.multivariate.structural.non_gaussian`.

References:
    Rigobon, R. (2003). Identification through heteroskedasticity. *Review
    of Economics and Statistics*, 85(4), 777-792.

    Lanne, M., & Lütkepohl, H. (2008). Identifying monetary policy shocks via
    changes in volatility. *Journal of Money, Credit and Banking*, 40(6),
    1131-1149.

    Lewis, D. J. (2021). Identifying shocks via time-varying volatility.
    *Review of Economic Studies*, 88(6), 3086-3124.

Example:
    One shock triples its standard deviation halfway through the sample, the
    other does not. Declaring the split recovers both impact columns from
    the variance shift alone, and the diagnostics carry the ratios that did
    the identifying:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
    >>> eps = rng.standard_normal((402, 2))
    >>> eps[201:, 0] *= 3.0
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
    >>> res = VAR(y, order=1).fit()
    >>> labels = ["calm"] * 200 + ["volatile"] * 201
    >>> svar = HeteroskedasticSVAR(res, labels, shock_names=("shifted", "steady")).identify()
    >>> svar.shock_names, dict(svar.diagnostics)["Identification"]
    (('shifted', 'steady'), 'distinct ratios')
    >>> line = dict(svar.diagnostics)["Variances in 'volatile' vs 'calm'"]
    >>> ratios = [float(v) for v in line.split(", ")]
    >>> bool(8.0 < ratios[0] < 11.0), bool(0.8 < ratios[1] < 1.2)
    (True, True)
    >>> unit = lambda v: v / np.linalg.norm(v)
    >>> bool(np.abs(unit(svar.impact[:, 1]) - unit(B[:, 1])).max() < 0.05)
    True
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ...engine._core import (
    ClosedSystemResult,
    _lower_cholesky,
    _validate_regimes,
)
from ...engine._internals import _CoDiagonalObjective, _IdentificationModel, _solve
from ...exceptions import NumericalError, SpecificationError
from .zero_restrictions import SVARResult

__all__ = ["HeteroskedasticSVAR"]


class HeteroskedasticSVAR(_IdentificationModel[SVARResult]):
    r"""Identification from declared variance regimes, Rigobon (2003).

    If the structural shocks' variances shift across regimes while the
    impact matrix does not,

    .. math::

       \Sigma_m = B \Lambda_m B', \qquad \Lambda_m \text{ diagonal},
       \qquad m = 1, \dots, M,

    then the regime covariances jointly pin down :math:`B` up to column
    order and sign, with no zero, sign pattern or instrument. Normalizing
    :math:`\Lambda_1 = I`, write :math:`\Sigma_1 = L L'`; the whitened
    second-regime covariance :math:`L^{-1} \Sigma_2 L^{-\top} = Q \Lambda_2
    Q'` is symmetric, its eigendecomposition gives the rotation :math:`Q`,
    and :math:`B = L Q`. The rotation is unique exactly when the diagonal of
    :math:`\Lambda_2` -- the variance ratios -- is distinct. With more than
    two regimes :math:`B` is over-identified: the two-regime solution is
    refined by joint approximate diagonalization, and the residual
    off-diagonal energy is the data's verdict on the constant-impact
    assumption.

    Shocks are normalized to unit variance in the *first* regime -- first in
    order of label appearance -- and ordered by descending variance ratio in
    the second, so the first column is the shock whose volatility shifted
    most. The per-regime relative variances are reported in the summary; they
    are the scheme's entire empirical content, and how far apart they sit is
    how strongly identified the model is.

    Note:
        The identifying assumption is the one the scheme's name hides in
        plain sight: the impact matrix is *constant* across the declared
        regimes and only the shock variances move. What the data cannot
        supply is meaning. The recovered shocks are labelled by their
        variance behaviour, and the default names are ``shock1 ... shockk``
        rather than the variable names on purpose: attaching "monetary" to
        a column is a claim the user must argue from outside the model, and
        the summary says so. Two shocks whose variances shift by the same
        factor are indistinguishable within their span; an exact tie is
        refused, and a separation within sampling noise is flagged ``WEAK``
        in the diagnostics.

    Attributes:
        _source: The closed reduced-form result being identified.
        _regime_labels: The regime labels in order of first appearance.
        _assignment: The ``(nobs,)`` integer regime of each residual row.
        _labels: The shock labels, in descending-ratio column order.

    See Also:
        * :class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
          -- the complete result returned.
        * :mod:`~cultivars.multivariate.structural.stochastic_volatility` --
          the same idea with the variance path estimated rather than
          declared.
        * :mod:`~cultivars.multivariate.structural.non_gaussian` --
          identification from higher moments instead of second-moment
          shifts.

    References:
        Rigobon, R. (2003). Identification through heteroskedasticity.
        *Review of Economics and Statistics*, 85(4), 777-792.

        Lanne, M., & Lütkepohl, H. (2008). Identifying monetary policy shocks
        via changes in volatility. *Journal of Money, Credit and Banking*,
        40(6), 1131-1149.

        Lewis, D. J. (2021). Identifying shocks via time-varying volatility.
        *Review of Economic Studies*, 88(6), 3086-3124.

    Example:
        A bivariate system whose first structural shock triples its standard
        deviation halfway through the sample. The declared split recovers
        both impact columns up to scale, the shock that shifted comes first,
        and the first regime's covariance factorizes exactly:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
        >>> eps = rng.standard_normal((402, 2))
        >>> eps[201:, 0] *= 3.0
        >>> y = np.zeros((402, 2))
        >>> for t in range(1, 402):
        ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("a", "b")).fit()
        >>> labels = ["calm"] * 200 + ["volatile"] * (res.resid.shape[0] - 200)
        >>> svar = HeteroskedasticSVAR(res, labels).identify()
        >>> svar.is_complete, svar.shock_names, dict(svar.diagnostics)["Identification"]
        (True, ('shock1', 'shock2'), 'distinct ratios')
        >>> dict(svar.diagnostics)["Variances in 'volatile' vs 'calm'"]
        '9.902, 0.941'
        >>> unit = lambda v: v / np.linalg.norm(v)
        >>> bool(np.abs(unit(svar.impact[:, 0]) - unit(B[:, 0])).max() < 0.03)
        True
        >>> calm = res.resid[:200]
        >>> bool(np.allclose(svar.impact @ svar.impact.T, calm.T @ calm / 200))
        True
    """

    __slots__ = ("_assignment", "_labels", "_regime_labels")

    def __init__(
        self,
        result: ClosedSystemResult,
        regimes: npt.ArrayLike,
        *,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system and the regime assignment.

        Args:
            result: The fitted closed reduced-form result to identify.
            regimes: One regime label per residual row of the result -- the
                effective sample, exactly as a proxy's instrument rows index
                it. Two or more regimes, each with enough observations to
                estimate a covariance.
            shock_names: One label per shock column, in descending-ratio
                order. Defaults to ``shock1 ... shockk``, deliberately not the
                variable names: these shocks are statistical objects until
                the user argues otherwise.

        Raises:
            SpecificationError: If the result is not a closed system, the
                regime assignment is malformed, or a regime is too short to
                estimate a covariance.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1).fit()
            >>> labels = ["a"] * 60 + ["b"] * 60
            >>> model = HeteroskedasticSVAR(res, labels, shock_names=("u", "v"))
            >>> model.regime_labels, model.n_regimes, model._labels
            (('a', 'b'), 2, ('u', 'v'))
            >>> HeteroskedasticSVAR(res, labels[:10])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: regimes must assign one label per residual ...
            >>> HeteroskedasticSVAR(res, ["a"] * 120)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: heteroskedasticity identifies through ...
            >>> HeteroskedasticSVAR(res, ["a"] * 119 + ["b"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: regime 'b' has 1 observations, too few ...
            >>> HeteroskedasticSVAR(res, labels, shock_names=("u",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: shock_names must be 2 unique labels; ...
        """
        super().__init__(result)
        k = self.k_endog
        nobs_resid = int(np.asarray(result.resid).shape[0])
        self._regime_labels, self._assignment = _validate_regimes(regimes, nobs=nobs_resid)
        for position, label in enumerate(self._regime_labels):
            count = int(np.sum(self._assignment == position))
            if count < k + 1:
                raise SpecificationError(
                    f"regime {label!r} has {count} observations, too few to "
                    f"estimate a {k}-variable covariance; it needs at least "
                    f"{k + 1}."
                )
        if shock_names is None:
            self._labels = tuple(f"shock{j + 1}" for j in range(k))
        else:
            resolved = tuple(str(name) for name in shock_names)
            if len(resolved) != k or len(set(resolved)) != k:
                raise SpecificationError(f"shock_names must be {k} unique labels; got {resolved}.")
            self._labels = resolved

    @property
    def regime_labels(self) -> tuple[str, ...]:
        """Regime labels, in order of first appearance.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1).fit()
            >>> HeteroskedasticSVAR(res, ["late"] * 60 + ["early"] * 60).regime_labels
            ('late', 'early')
        """
        return self._regime_labels

    @property
    def n_regimes(self) -> int:
        """Number of declared regimes.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1).fit()
            >>> HeteroskedasticSVAR(res, ["a"] * 40 + ["b"] * 40 + ["c"] * 40).n_regimes
            3
        """
        return len(self._regime_labels)

    def _regime_covariances(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Per-regime second moments of the residuals.

        Returns:
            One ``(k, k)`` matrix per regime in label order, each the
            uncentered second moment ``resid.T @ resid / n_m`` of that
            regime's residual rows.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1).fit()
            >>> model = HeteroskedasticSVAR(res, ["a"] * 60 + ["b"] * 60)
            >>> first, second = model._regime_covariances()
            >>> first.shape, bool(np.allclose(first, res.resid[:60].T @ res.resid[:60] / 60))
            ((2, 2), True)
        """
        resid = np.asarray(self.source.resid, dtype=np.float64)
        out: list[npt.NDArray[np.float64]] = []
        for position in range(self.n_regimes):
            block = resid[self._assignment == position]
            out.append(block.T @ block / block.shape[0])
        return tuple(out)

    def identify(self) -> SVARResult:
        """Recover the impact matrix from the variance shifts.

        Whitens every later regime's covariance by the first regime's
        Cholesky factor, takes the eigenvectors of the whitened second
        regime, refines them by joint diagonalization when there are more
        than two regimes, orders the columns by descending ratio, and signs
        each column so its largest entry is positive.

        Returns:
            The complete structural result, shock columns in descending order
            of second-regime variance ratio, unit variance in the first
            regime, with the per-regime variance ratios, the minimum ratio
            separation, an identification verdict and -- beyond two regimes
            -- the co-diagonalization residual in its diagnostics.

        Raises:
            NumericalError: If a regime covariance is not positive definite,
                or two variance ratios coincide and the shocks sharing them
                are unidentified.

        Example:
            A system whose shocks both triple their standard deviation across
            the split carries no identifying information -- the ratios tie
            up to sampling noise -- and the diagnostics say so; with three
            regimes the over-identified fit reports its residual:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> eps = rng.standard_normal((402, 2))
            >>> eps[201:] *= 3.0
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
            >>> labels = ["calm"] * 200 + ["volatile"] * 201
            >>> tied = HeteroskedasticSVAR(VAR(y, order=1).fit(), labels).identify()
            >>> dict(tied.diagnostics)["Identification"]
            'WEAK: separation within sampling noise'
            >>> eps = rng.standard_normal((402, 2))
            >>> eps[131:261, 0] *= 3.0
            >>> eps[261:, 1] *= 2.5
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
            >>> three = ["a"] * 130 + ["b"] * 130 + ["c"] * 141
            >>> svar = HeteroskedasticSVAR(VAR(y, order=1).fit(), three, shock_names=("u", "v"))
            >>> report = dict(svar.identify().diagnostics)
            >>> report["Regimes"], report["Identification"], "Co-diagonalization residual" in report
            ('3', 'distinct ratios', True)
        """
        k = self.k_endog
        covariances = self._regime_covariances()
        factor = _lower_cholesky(
            covariances[0], f"the {self._regime_labels[0]!r} regime covariance"
        )
        whitened = tuple(
            np.linalg.solve(factor, np.linalg.solve(factor, sigma).T).T for sigma in covariances[1:]
        )
        ratios, rotation = np.linalg.eigh((whitened[0] + whitened[0].T) / 2.0)
        order = np.argsort(ratios)[::-1]
        ratios = ratios[order]
        rotation = rotation[:, order]

        residual = 0.0
        if len(whitened) > 1:
            objective = _CoDiagonalObjective(
                targets=tuple(rotation.T @ target @ rotation for target in whitened)
            )
            refinement, residual = _solve(objective)
            rotation = rotation @ refinement
            ratios = np.diagonal(rotation.T @ whitened[0] @ rotation).copy()
            order = np.argsort(ratios)[::-1]
            ratios = ratios[order]
            rotation = rotation[:, order]

        gaps = np.abs(np.diff(ratios)) / (1.0 + np.abs(ratios[:-1]))
        separation = float(gaps.min()) if gaps.size else np.inf
        counts = np.bincount(self._assignment, minlength=self.n_regimes)
        noise_scale = 4.0 * float(np.sqrt(2.0 / counts.min()))
        if separation < 1e-8:
            raise NumericalError(
                "two shocks' variance ratios coincide, so the rotation within "
                "their span is not identified: heteroskedasticity separates "
                "shocks only where their volatilities shifted by different "
                "factors. Merge or re-cut the regimes, or accept that these "
                "shocks need an economic restriction to tell apart."
            )

        impact = factor @ rotation
        for j in range(k):
            column = impact[:, j]
            if column[int(np.argmax(np.abs(column)))] < 0.0:
                impact[:, j] = -column

        variance_lines: list[tuple[str, str]] = []
        for position, label in enumerate(self._regime_labels[1:], start=1):
            diag = np.diagonal(rotation.T @ whitened[position - 1] @ rotation)
            variance_lines.append(
                (
                    f"Variances in {label!r} vs {self._regime_labels[0]!r}",
                    ", ".join(f"{value:.3f}" for value in diag),
                )
            )
        diagnostics: list[tuple[str, str]] = [
            ("Regimes", f"{self.n_regimes}"),
            *variance_lines,
            ("Min ratio separation", f"{separation:.4f}"),
            (
                "Identification",
                "WEAK: separation within sampling noise"
                if separation < noise_scale
                else "distinct ratios",
            ),
        ]
        if len(whitened) > 1:
            diagnostics.append(("Co-diagonalization residual", f"{residual:.3e}"))
        return SVARResult(
            source=self.source,
            impact=impact,
            shock_names=self._labels,
            scheme="heteroskedasticity",
            restriction=(
                "The impact matrix is assumed constant across the declared "
                f"regimes {self._regime_labels}, with only the shock "
                "variances shifting; that assumption is the entire "
                "identification. Shocks are unit-variance in the "
                f"{self._regime_labels[0]!r} regime, ordered by descending "
                "variance ratio, and labelled by their variance behavior "
                "rather than by economics -- an economic name for any of them "
                "is a claim to be argued from outside the model. Near-equal "
                "ratios mean weak identification; the separation is reported "
                "above."
            ),
            diagnostics=tuple(diagnostics),
        )
