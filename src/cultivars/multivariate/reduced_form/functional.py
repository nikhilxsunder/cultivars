# filepath: /src/cultivars/var/reduced_form/functional.py
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

"""The functional vector autoregression: dynamics for observations that are curves.

A yield curve, a demographic profile, an intraday volatility signature -- each
observation is a function evaluated on a grid, and modelling every grid point as
its own variable buys nothing but a singular covariance. The functional VAR
factors the problem instead: project each curve onto a small basis, run an
ordinary VAR on the basis scores, and map everything the VAR produces --
fitted values, forecasts, impulse responses -- back through the basis to curve
space.

The design is composition rather than inheritance, and deliberately so. The
factor dynamics *are* a :class:`~cultivars.var.reduced_form.VAR`, with every
coefficient table, diagnostic, and decomposition that implies, and the fitted
factor result rides on the functional result as ``.factors`` rather than being
re-wrapped method by method. What this module adds is exactly the part that is
functional: the projection, its quality, and the maps between curve space and
score space.

Three bases, one engine. Functional principal components are estimated from the
sample and are optimal in mean square for it; Nelson-Siegel is the three-factor
level-slope-curvature structure the term-structure literature runs on, with the
decay either fixed or profiled; B-splines are the local, shape-agnostic choice
when neither a data-driven nor a yield-curve basis fits the problem.

References:
    Bosq, D. (2000). *Linear Processes in Function Spaces*. Springer.
    Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.
    Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*.
        Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from ..._core import (
    FunctionalBasis,
    SummaryTable,
    Trend,
    _nelson_siegel_loadings,
    _projection_scores,
    _validate_curves,
    validate_choice,
)
from ..._internals import _SummaryMixin
from ...exceptions import DimensionError, NumericalError, SpecificationError
from .vector_autoregression import VAR, VARResult


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FunctionalVARResult(_SummaryMixin):
    """A fitted functional vector autoregression.

    Composition, stated plainly: :attr:`factors` is a complete
    :class:`~cultivars.var.reduced_form.VARResult` on the basis scores, and
    every factor-level question -- coefficients, standard errors, Granger
    causality, stability, residual diagnostics, factor impulse responses --
    is answered there, not re-exported here one method at a time. This object
    owns what the factor result cannot know: the basis, the projection's
    quality, and the maps back to curve space.

    Attributes:
        curves: The sample, one curve per row.
        grid: The evaluation points, one per column of :attr:`curves`.
        basis: Which basis family was used.
        basis_matrix: The ``(n_points, r)`` basis the scores live in.
        mean_curve: The curve subtracted before projection -- the sample mean
            under ``"fpca"``, zero under the fixed bases, whose level factor
            plays that role itself.
        scores: The ``(nobs, r)`` projected sample the factor VAR was fitted
            to.
        factors: The fitted factor VAR, carrying the entire reduced-form
            surface at factor level.
        explained_variance: Share of centered sample variance each retained
            component carries, under ``"fpca"``; ``None`` otherwise.
        decay: The Nelson-Siegel decay used, under ``"nelson-siegel"``;
            ``None`` otherwise.
    """

    curves: npt.NDArray[np.float64] = field(repr=False)
    grid: npt.NDArray[np.float64] = field(repr=False)
    basis: str
    basis_matrix: npt.NDArray[np.float64] = field(repr=False)
    mean_curve: npt.NDArray[np.float64] = field(repr=False)
    scores: npt.NDArray[np.float64] = field(repr=False)
    factors: VARResult = field(repr=False)
    explained_variance: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    decay: float | None = None

    @property
    def k_factors(self) -> int:
        """Number of basis functions."""
        return int(self.basis_matrix.shape[1])

    @property
    def n_points(self) -> int:
        """Number of grid points per curve."""
        return int(self.basis_matrix.shape[0])

    @property
    def nobs(self) -> int:
        """Number of curves in the sample."""
        return int(self.curves.shape[0])

    def projected_curves(self) -> npt.NDArray[np.float64]:
        """The sample as the basis sees it, over the full sample.

        Returns:
            An ``(nobs, n_points)`` array: each curve replaced by its
            projection onto the basis. The gap between this and
            :attr:`curves` is pure approximation error -- what the basis
            cannot represent -- before any dynamics enter.
        """
        return self.mean_curve + self.scores @ self.basis_matrix.T

    @property
    def reconstruction_rmse(self) -> float:
        """Root-mean-square projection error over all curves and grid points.

        The floor for everything downstream: no forecast or fitted value can
        be closer to the data than the basis allows the data to be to itself.
        """
        return float(np.sqrt(np.mean((self.curves - self.projected_curves()) ** 2)))

    def fitted_curves(self) -> npt.NDArray[np.float64]:
        """One-step conditional mean curves over the factor VAR's effective sample.

        Returns:
            An ``(nobs - p, n_points)`` array, the factor VAR's fitted values
            mapped through the basis.
        """
        return self.mean_curve + self.factors.fittedvalues @ self.basis_matrix.T

    def forecast_curves(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Deterministic multi-step curve forecasts from the end of the sample.

        Args:
            steps: Forecast horizon, at least one.

        Returns:
            An array of shape ``(steps, n_points)``.
        """
        return self.mean_curve + self.factors.forecast(steps) @ self.basis_matrix.T

    def irf_curves(
        self, horizon: int = 20, *, orthogonalized: bool = True
    ) -> npt.NDArray[np.float64]:
        """The curve's response over time to a one-unit factor shock.

        Args:
            horizon: Largest lead to return.
            orthogonalized: Whether to orthogonalize the factor shocks by the
                Cholesky factor of their innovation covariance, in the factor
                ordering -- with the same ordering caveat that carries.

        Returns:
            An array of shape ``(horizon + 1, n_points, k_factors)`` whose
            entry ``[h, :, j]`` is the whole curve's response at lead ``h`` to
            a shock in factor ``j``, the factor impulse response mapped
            through the basis.
        """
        psi = self.factors.irf(horizon, orthogonalized=orthogonalized)
        return np.einsum("pr,hrj->hpj", self.basis_matrix, psi)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        stability = self.factors.stability_check()
        label = {"fpca": "functional principal components"}.get(self.basis, self.basis)
        rows: list[tuple[str, ...]] = []
        for j, name in enumerate(self.factors.names):
            share = (
                f"{float(self.explained_variance[j]):.1%}"
                if self.explained_variance is not None
                else "-"
            )
            rows.append((name, share, f"{float(np.abs(self.scores[:, j]).mean()):.4f}"))
        notes = [
            f"Factor VAR stable: {self.factors.is_stable}   max |companion root| = "
            f"{stability.max_modulus:.4f}",
            "Factor-level inference -- coefficients, standard errors, Granger "
            "causality, diagnostics, factor impulse responses -- lives on "
            ".factors, a complete VAR result on the scores.",
            "No curve-level output can beat the reconstruction floor: the basis "
            "bounds how well the data can represent itself before any dynamics "
            "are estimated.",
        ]
        if self.decay is not None:
            notes.insert(2, f"Nelson-Siegel decay lambda = {self.decay:.6g}.")
        return SummaryTable(
            title=f"FunctionalVAR({self.factors.order}) Results",
            metadata=(
                ("Basis", label),
                ("Factors", f"{self.k_factors}"),
                ("Grid points", f"{self.n_points}"),
                ("Curves", f"{self.nobs}"),
                ("Order", f"{self.factors.order}"),
                ("Trend", self.factors.trend),
                ("Reconstruction RMSE", f"{self.reconstruction_rmse:.6f}"),
                ("Factor log-likelihood", f"{self.factors.llf:.3f}"),
            ),
            columns=("factor", "variance share", "mean |score|"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class FunctionalVAR:
    """A vector autoregression for observations that are curves on a grid.

    Project, fit, map back. Each curve is reduced to its coordinates in a
    small basis, an ordinary VAR is estimated on those coordinates, and the
    curve-level surface -- fitted curves, forecast curves, curve impulse
    responses -- is the factor surface mapped through the basis. The choice of
    basis is the modelling decision, and the three on offer answer three
    different situations:

    ``"fpca"`` estimates the basis from the sample: the leading eigenfunctions
    of the empirical covariance, optimal in mean square for these data, with
    the component count chosen explicitly or by a variance-share target.

    ``"nelson-siegel"`` fixes the basis to level, slope, and curvature, which
    is the Diebold-Li dynamic form of the term-structure literature. The grid
    is then a vector of maturities and must be strictly positive; the decay is
    supplied, or profiled over the curvature-peak candidates the grid itself
    implies.

    ``"bspline"`` fixes a local polynomial basis with a chosen number of
    degrees of freedom -- the shape-agnostic option when neither the data-driven
    nor the yield-curve structure fits.

    Args:
        curves: The ``(nobs, n_points)`` panel, one curve per row.
        grid: The ``(n_points,)`` evaluation points. Defaults to ``0, 1, ...``;
            required in substance for ``"nelson-siegel"``, where it carries the
            maturities.
        order: Autoregressive order of the factor VAR.
        basis: One of ``"fpca"``, ``"nelson-siegel"``, ``"bspline"``.
        n_components: Retained components under ``"fpca"``. ``None`` keeps the
            smallest count explaining at least 95% of centered sample variance.
        decay: The Nelson-Siegel ``lambda``. ``None`` profiles it by
            reconstruction error over the candidates ``1.79 / tau`` for each
            grid maturity, the decay that puts the curvature peak at ``tau``.
        df: Spline degrees of freedom under ``"bspline"``; defaults to
            ``min(8, n_points)``.
        degree: Spline degree under ``"bspline"``.
        trend: Deterministic terms of the factor VAR.

    Raises:
        SpecificationError: If the basis, its parameters, or the grid are
            malformed for the chosen basis.
        DimensionError: If the shapes disagree, or the sample is too short for
            the factor VAR the projection implies.
        NumericalError: If the curves are non-finite or the basis is
            rank-deficient on the grid.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> grid = np.linspace(0.0, 1.0, 25)
        >>> level = np.cumsum(rng.standard_normal(80))[:, np.newaxis] * 0.1
        >>> curves = level + np.sin(np.pi * grid) + 0.01 * rng.standard_normal((80, 25))
        >>> res = FunctionalVAR(curves, grid, order=1, basis="fpca").fit()
        >>> res.forecast_curves(3).shape
        (3, 25)
    """

    __slots__ = (
        "_basis",
        "_curves",
        "_decay",
        "_degree",
        "_df",
        "_grid",
        "_n_components",
        "_order",
        "_trend",
    )

    def __init__(
        self,
        curves: npt.ArrayLike,
        grid: npt.ArrayLike | None = None,
        *,
        order: int,
        basis: FunctionalBasis = "fpca",
        n_components: int | None = None,
        decay: float | None = None,
        df: int | None = None,
        degree: int = 3,
        trend: Trend = "c",
    ) -> None:
        """Validate the curves, the grid, and the basis specification."""
        self._curves = _validate_curves(curves)
        n_points = self._curves.shape[1]
        self._grid = (
            np.arange(n_points, dtype=np.float64)
            if grid is None
            else np.asarray(grid, dtype=np.float64).ravel()
        )
        if self._grid.shape[0] != n_points:
            raise DimensionError(
                f"grid must have one entry per curve column ({n_points}); got "
                f"{self._grid.shape[0]}."
            )
        if not np.all(np.isfinite(self._grid)):
            raise NumericalError("grid must be finite.")
        if np.any(np.diff(self._grid) <= 0):
            raise SpecificationError("grid must be strictly increasing.")
        self._basis: str = validate_choice(basis, FunctionalBasis, "basis")
        self._order = int(order)
        self._trend: Trend = validate_choice(trend, Trend, "trend")

        if n_components is not None and (int(n_components) != n_components or n_components < 1):
            raise SpecificationError(f"n_components must be an integer >= 1; got {n_components!r}.")
        self._n_components = None if n_components is None else int(n_components)
        if decay is not None and (not np.isfinite(decay) or decay <= 0):
            raise SpecificationError(f"decay must be a positive number; got {decay!r}.")
        self._decay = None if decay is None else float(decay)
        if int(degree) != degree or degree < 1:
            raise SpecificationError(f"degree must be an integer >= 1; got {degree!r}.")
        self._degree = int(degree)
        resolved_df = min(8, n_points) if df is None else df
        if int(resolved_df) != resolved_df or resolved_df < self._degree + 1:
            raise SpecificationError(
                f"df must be an integer >= degree + 1 = {self._degree + 1}; got {df!r}."
            )
        if resolved_df > n_points:
            raise SpecificationError(
                f"df ({resolved_df}) cannot exceed the number of grid points ({n_points})."
            )
        self._df = int(resolved_df)
        if self._basis == "nelson-siegel" and np.any(self._grid <= 0):
            raise SpecificationError(
                "nelson-siegel needs a strictly positive grid: the grid carries "
                "the maturities, and the loadings are undefined at zero."
            )

    @property
    def basis(self) -> str:
        """The basis family."""
        return self._basis

    @property
    def order(self) -> int:
        """Autoregressive order of the factor VAR."""
        return self._order

    def _fpca_basis(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Estimate the principal-component basis from the centered sample.

        Returns:
            The basis matrix, the mean curve, and the retained components'
            variance shares.
        """
        mean = self._curves.mean(axis=0)
        centered = self._curves - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        shares = singular**2 / float(np.sum(singular**2))
        cap = int(np.sum(singular > singular[0] * 1e-10))
        if self._n_components is None:
            retained = int(np.searchsorted(np.cumsum(shares), 0.95) + 1)
        else:
            retained = self._n_components
        if retained > cap:
            raise SpecificationError(
                f"n_components ({retained}) exceeds the sample's numerical rank ({cap})."
            )
        return vt[:retained].T.copy(), mean, shares[:retained].copy()

    def _nelson_siegel_basis(self) -> tuple[npt.NDArray[np.float64], float]:
        """Fix or profile the Nelson-Siegel loadings.

        Returns:
            The loading matrix and the decay it was built at.
        """
        if self._decay is not None:
            return _nelson_siegel_loadings(self._grid, self._decay), self._decay
        best_basis: npt.NDArray[np.float64] | None = None
        best_decay = 0.0
        best_sse = np.inf
        for tau in self._grid:
            candidate = 1.79 / float(tau)
            loadings = _nelson_siegel_loadings(self._grid, candidate)
            scores = _projection_scores(self._curves, loadings)
            sse = float(np.sum((self._curves - scores @ loadings.T) ** 2))
            if sse < best_sse:
                best_sse, best_basis, best_decay = sse, loadings, candidate
        assert best_basis is not None
        return best_basis, best_decay

    def _bspline_basis(self) -> npt.NDArray[np.float64]:
        """Build the clamped B-spline design on the grid."""
        inner = self._df - self._degree - 1
        low, high = float(self._grid[0]), float(self._grid[-1])
        interior = (
            np.quantile(self._grid, np.linspace(0.0, 1.0, inner + 2)[1:-1])
            if inner
            else np.empty(0)
        )
        knots = np.concatenate(
            [np.full(self._degree + 1, low), interior, np.full(self._degree + 1, high)]
        )
        return np.asarray(
            BSpline.design_matrix(self._grid, knots, self._degree).toarray(),
            dtype=np.float64,
        )

    def fit(self) -> FunctionalVARResult:
        """Project the curves and estimate the factor VAR.

        Returns:
            The fitted result, with the complete factor-level surface on its
            ``factors`` attribute.
        """
        explained: npt.NDArray[np.float64] | None = None
        decay: float | None = None
        n_points = self._curves.shape[1]
        if self._basis == "fpca":
            matrix, mean, explained = self._fpca_basis()
            scores = (self._curves - mean) @ matrix
            names = tuple(f"pc{j + 1}" for j in range(matrix.shape[1]))
        elif self._basis == "nelson-siegel":
            matrix, decay = self._nelson_siegel_basis()
            mean = np.zeros(n_points, dtype=np.float64)
            scores = _projection_scores(self._curves, matrix)
            names = ("level", "slope", "curvature")
        else:
            matrix = self._bspline_basis()
            mean = np.zeros(n_points, dtype=np.float64)
            scores = _projection_scores(self._curves, matrix)
            names = tuple(f"b{j + 1}" for j in range(matrix.shape[1]))
        factors = VAR(scores, order=self._order, trend=self._trend, names=names).fit()
        return FunctionalVARResult(
            curves=self._curves,
            grid=self._grid,
            basis=self._basis,
            basis_matrix=matrix,
            mean_curve=mean,
            scores=scores,
            factors=factors,
            explained_variance=explained,
            decay=decay,
        )
