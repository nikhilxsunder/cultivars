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

"""Identification through heteroskedasticity: the data identify, the user labels.

Every other scheme in this package buys identification with an economic
restriction someone must defend. This one buys it with a statistical fact: if
the structural shocks' variances shift across declared regimes while the
impact matrix stays constant, the regime covariances ``Sigma_m = B Lambda_m
B'`` jointly pin down ``B`` up to column order and sign, and no zero, no sign
pattern, no instrument is needed -- Rigobon (2003). What the data cannot
supply is meaning: the recovered shocks are labelled by their variance
behavior, not by economics, and attaching names like "monetary" to them is a
claim the user must argue from outside the model, which the summary says
plainly.

With two regimes the solution is closed form -- a symmetric eigenvalue
problem in the whitened second-regime covariance. With more, the impact
matrix is over-identified: no single rotation exactly diagonalizes every
regime in sample, the estimate comes from joint approximate diagonalization
warm-started at the two-regime solution, and the residual off-diagonal energy
is reported as the data's verdict on the constant-impact assumption.

The identification condition is empirically checkable, and checked: the
variance ratios must be distinct. Two shocks whose variances shift by the
same factor are indistinguishable within their span, so near-equal ratios are
weak identification and an exact tie is a refusal.

References:
    Rigobon, R. (2003). Identification through heteroskedasticity. *Review of
        Economics and Statistics*, 85(4), 777-792.
    Lanne, M., & Lutkepohl, H. (2008). Identifying monetary policy shocks via
        changes in volatility. *Journal of Money, Credit and Banking*, 40(6),
        1131-1149.
    Lewis, D. J. (2021). Identifying shocks via time-varying volatility.
        *Review of Economic Studies*, 88(6), 3086-3124.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..._core import (
    ClosedSystemResult,
    _lower_cholesky,
    _validate_regimes,
)
from ..._internals import _CoDiagonalObjective, _IdentificationModel, _solve
from ...exceptions import NumericalError, SpecificationError
from .zero_restrictions import SVARResult


class HeteroskedasticSVAR(_IdentificationModel[SVARResult]):
    """Identification from declared variance regimes, Rigobon (2003).

    The identifying assumption is the one the scheme's name hides in plain
    sight: the impact matrix is *constant* across the declared regimes, and
    only the shock variances move. Under it, whitening the later regimes'
    covariances by the first regime's Cholesky factor turns identification
    into diagonalization: the rotation that diagonalizes them is unique up to
    column order and sign exactly when the variance ratios are distinct, and
    the impact matrix is the factor times that rotation.

    Shocks are normalized to unit variance in the *first* regime -- first in
    order of label appearance -- and ordered by descending variance ratio in
    the second, so the first column is the shock whose volatility shifted
    most. The per-regime relative variances are reported in the summary; they
    are the scheme's entire empirical content, and how far apart they sit is
    how strongly identified the model is.

    Args:
        result: The fitted closed reduced-form result to identify.
        regimes: One regime label per residual row of the result -- the
            effective sample, exactly as a proxy's instrument rows index it.
            Two or more regimes, each with enough observations to estimate a
            covariance.
        shock_names: One label per shock column, in descending-ratio order.
            Defaults to ``shock1 ... shockk``, deliberately not the variable
            names: these shocks are statistical objects until the user argues
            otherwise.

    Raises:
        SpecificationError: If the result is not a closed system, the regime
            assignment is malformed, or a regime is too short to estimate a
            covariance.
        NumericalError: If a regime covariance is not positive definite, or
            two variance ratios coincide -- in which case the shocks sharing
            the ratio are not identified.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> shocks = rng.standard_normal((400, 2))
        >>> shocks[200:, 0] *= 3.0
        >>> y = np.zeros((400, 2))
        >>> for t in range(1, 400):
        ...     y[t] = 0.4 * y[t - 1] + shocks[t]
        >>> res = VAR(y, order=1).fit()
        >>> labels = ["calm"] * (res.resid.shape[0] // 2) + ["volatile"] * (
        ...     res.resid.shape[0] - res.resid.shape[0] // 2
        ... )
        >>> svar = HeteroskedasticSVAR(res, labels).identify()
        >>> svar.is_complete
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
        """Validate the source system and the regime assignment."""
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
        """Regime labels, in order of first appearance."""
        return self._regime_labels

    @property
    def n_regimes(self) -> int:
        """Number of declared regimes."""
        return len(self._regime_labels)

    def _regime_covariances(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Per-regime second moments of the residuals."""
        resid = np.asarray(self.source.resid, dtype=np.float64)
        out: list[npt.NDArray[np.float64]] = []
        for position in range(self.n_regimes):
            block = resid[self._assignment == position]
            out.append(block.T @ block / block.shape[0])
        return tuple(out)

    def identify(self) -> SVARResult:
        """Recover the impact matrix from the variance shifts.

        Returns:
            The complete structural result, shock columns in descending order
            of second-regime variance ratio, unit variance in the first
            regime.

        Raises:
            NumericalError: If a regime covariance is not positive definite,
                or two variance ratios coincide and the shocks sharing them
                are unidentified.
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
