# filepath: /src/cultivars/multivariate/reduced_form/functional.py
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
r"""The functional vector autoregression: dynamics for observations that are curves.

A yield curve, a demographic profile, an intraday volatility signature -- each
observation is a function evaluated on a grid, and modelling every grid point as
its own variable buys nothing but a singular covariance. The functional VAR
factors the problem instead: write each curve as a mean plus coordinates in a
small basis,

.. math::

   y_t(s) \approx \bar y(s) + \sum_{j=1}^{r} \xi_{tj}\, \phi_j(s),
   \qquad
   \xi_t = \sum_{i=1}^{p} A_i\, \xi_{t-i} + c + u_t,

run an ordinary VAR on the score vector :math:`\xi_t`, and map everything the
VAR produces -- fitted values, forecasts, impulse responses -- back through the
basis matrix :math:`\Phi = [\phi_j(s_k)]` to curve space. This is the
Diebold-Li dynamic Nelson-Siegel idea in general form: the operator
autoregression of Bosq, truncated to a finite basis so that it becomes a VAR
the rest of the package already knows how to estimate, test and propagate.

Two commitments shape the surface. First, the design is composition rather
than inheritance, and deliberately so. The factor dynamics *are* a
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`, with
every coefficient table, diagnostic, Granger test and decomposition that
implies, and the fitted factor result rides on the functional result as
``.factors`` rather than being re-wrapped method by method. What this module
adds is exactly the part that is functional -- the projection, its quality,
and the maps between curve space and score space -- and it names the floor
that the projection puts under every curve-level number: no fitted or
forecast curve can beat the reconstruction error of the basis itself.
Second, the basis is the modelling decision and is left to the caller.
Functional principal components are estimated from the sample and are
optimal in mean square for it; Nelson-Siegel fixes the three-factor
level-slope-curvature structure the term-structure literature runs on, with
the decay either fixed or profiled; B-splines are the local, shape-agnostic
choice when neither a data-driven nor a yield-curve basis fits the problem.
The three share one estimation path and one result type, so a comparison
across bases is a comparison of reconstruction floors and factor fits, not of
machinery.

Layout. :class:`FunctionalVAR` validates the panel through
``_validate_curves`` and the grid in place, builds the basis in
``_fpca_basis`` (a thin SVD of the centered panel), ``_nelson_siegel_basis``
(``_nelson_siegel_loadings`` from ``_core._polynomials`` at a fixed or
profiled decay) or ``_bspline_basis`` (a clamped :class:`scipy.interpolate.BSpline`
design at quantile knots), projects the curves with ``_projection_scores``
from ``_core``, and fits a
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR` to
the scores. :class:`FunctionalVARResult` takes its summary from
``_SummaryMixin`` and otherwise holds the basis, the mean curve, the scores
and the factor result, with the curve-level maps as thin products with
:math:`\Phi`. The dedicated yield-curve models that fix the Nelson-Siegel
structure and estimate the decay jointly are in
:mod:`~cultivars.multivariate.reduced_form.term_structure`; the factor
models that estimate loadings from a wide panel rather than a grid are in
:mod:`~cultivars.multivariate.large_dim`.

References:
    Bosq, D. (2000). *Linear Processes in Function Spaces*. Springer.

    Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
    government bond yields. *Journal of Econometrics*, 130(2), 337-364.

    Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis*.
    Springer.

    Kokoszka, P., & Reimherr, M. (2017). *Introduction to Functional Data
    Analysis*. Chapman and Hall/CRC.

Example:
    Yield curves on ten maturities generated from level, slope and curvature
    factors at decay 0.6. The Nelson-Siegel basis profiles the decay back to
    the grid candidate nearest 0.6 and names the factors; three principal
    components reach the same reconstruction floor without the names. A
    level shock moves every maturity by one at impact, and the factor VAR's
    own surface answers the factor-level questions:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    >>> x = 0.6 * maturities
    >>> slope, curvature = (1 - np.exp(-x)) / x, (1 - np.exp(-x)) / x - np.exp(-x)
    >>> loadings = np.column_stack([np.ones(10), slope, curvature])
    >>> factors = np.zeros((240, 3))
    >>> for t in range(1, 240):
    ...     factors[t] = [0.1, -0.15, 0.1] + [0.98, 0.9, 0.8] * factors[t - 1]
    ...     factors[t] = factors[t] + rng.standard_normal(3) * [0.2, 0.3, 0.4]
    >>> yields = factors @ loadings.T + 0.02 * rng.standard_normal((240, 10))
    >>> ns = FunctionalVAR(yields, maturities, order=1, basis="nelson-siegel").fit()
    >>> pc = FunctionalVAR(yields, maturities, order=1, basis="fpca", n_components=3).fit()
    >>> round(ns.decay, 3), ns.factors.names
    (0.597, ('level', 'slope', 'curvature'))
    >>> bool(abs(ns.reconstruction_rmse - pc.reconstruction_rmse) < 0.001)
    True
    >>> bool(np.allclose(ns.irf_curves(0, orthogonalized=False)[0, :, 0], 1.0))
    True
    >>> ns.factors.coefficients[0].diagonal().round(1)
    array([1. , 0.8, 0.8])
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

__all__ = ["FunctionalVAR", "FunctionalVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FunctionalVARResult(_SummaryMixin):
    r"""A fitted functional vector autoregression.

    Each curve :math:`y_t(\cdot)`, observed on a grid of :math:`n` points, is
    written as a mean curve plus a combination of :math:`r` basis functions,

    .. math::

       y_t(s) \approx \bar y(s) + \sum_{j=1}^{r} \xi_{tj}\, \phi_j(s),
       \qquad
       \xi_t = \sum_{i=1}^{p} A_i\, \xi_{t-i} + c + u_t,

    and the dynamics live entirely in the score vector :math:`\xi_t`. The
    curve-level surface is the score surface mapped through the basis matrix
    :math:`\Phi = [\phi_j(s_k)]`: fitted curves are :math:`\bar y + \hat\xi_t
    \Phi'`, forecast curves are the factor forecast through :math:`\Phi'`,
    and the response of the whole curve to a factor shock is the factor
    impulse response through :math:`\Phi`.

    Composition, stated plainly: :attr:`factors` is a complete
    :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
    on the basis scores, and every factor-level question -- coefficients,
    standard errors, Granger causality, stability, residual diagnostics,
    factor impulse responses -- is answered there, not re-exported here one
    method at a time. This object owns what the factor result cannot know:
    the basis, the projection's quality, and the maps back to curve space.

    Note:
        Two floors bound every curve-level number. The projection floor,
        :attr:`reconstruction_rmse`, is how far the data sit from their own
        best representation in the basis before any dynamics enter; no
        fitted or forecast curve can beat it. The factor VAR's own fit is
        the second, and the one-step curve error is the two combined. Under
        ``"fpca"`` the basis is estimated from these curves, so the floor is
        in-sample optimal and the retained share of variance is reported;
        under the fixed bases it is whatever the chosen functions allow, and
        :attr:`explained_variance` is ``None`` because the question has no
        analogue. The basis is a mean-square choice, not a dynamic one: a
        variance-share target on level-dominated curves (yields, say) can
        keep a single level component and discard the slope and curvature
        movements the dynamics are about, which is what ``n_components`` is
        for.

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

    See Also:
        * :class:`FunctionalVAR` -- the model whose ``fit()`` returns this
          record and chooses the basis.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the type of :attr:`factors`, where all factor-level inference
          lives.
        * :mod:`~cultivars.multivariate.reduced_form.term_structure` -- the
          yield-curve models that fix the Nelson-Siegel structure and add
          the pricing restrictions this basis does not impose.

    References:
        Bosq, D. (2000). *Linear Processes in Function Spaces*. Springer.

        Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.

        Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data
        Analysis*. Springer.

    Example:
        Curves that share a fixed shape and move through a random-walk level
        and an autoregressive tilt. Two principal components carry the two
        movements, the reconstruction floor is a fraction of the noise, and
        the curve forecast, the fitted curves and the curve impulse
        responses are the factor surface pushed through the basis:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> grid = np.linspace(0.0, 1.0, 25)
        >>> level = 0.1 * np.cumsum(rng.standard_normal(80))
        >>> tilt = np.zeros(80)
        >>> for t in range(1, 80):
        ...     tilt[t] = 0.7 * tilt[t - 1] + 0.1 * rng.standard_normal()
        >>> shape = np.sin(np.pi * grid)
        >>> curves = level[:, None] + tilt[:, None] * (grid - 0.5) + shape
        >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
        >>> res = FunctionalVAR(curves, grid, order=1, basis="fpca", n_components=2).fit()
        >>> res.k_factors, res.n_points, res.nobs, res.factors.names
        (2, 25, 80, ('pc1', 'pc2'))
        >>> res.explained_variance.round(3), bool(res.reconstruction_rmse < 0.02)
        (array([0.991, 0.008]), True)
        >>> res.fitted_curves().shape, res.forecast_curves(3).shape, res.irf_curves(4).shape
        ((79, 25), (3, 25), (5, 25, 2))
        >>> bool(np.allclose(res.scores, (curves - res.mean_curve) @ res.basis_matrix))
        True
    """

    curves: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, n_points)`` sample, one curve per row, as validated. Kept out of the repr."""

    grid: npt.NDArray[np.float64] = field(repr=False)
    """The ``(n_points,)`` strictly increasing evaluation points. Kept out of the repr."""

    basis: str
    """The basis family: ``"fpca"``, ``"nelson-siegel"`` or ``"bspline"``."""

    basis_matrix: npt.NDArray[np.float64] = field(repr=False)
    r"""The ``(n_points, r)`` matrix :math:`\Phi` of basis functions on the grid.

    Kept out of the repr.
    """

    mean_curve: npt.NDArray[np.float64] = field(repr=False)
    """The ``(n_points,)`` curve subtracted before projection. Kept out of the repr."""

    scores: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, r)`` basis coordinates the factor VAR was fitted to. Kept out of the repr."""

    factors: VARResult = field(repr=False)
    """The fitted factor VAR on :attr:`scores`, named by the basis. Kept out of the repr."""

    explained_variance: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Per-component shares of centered variance under ``"fpca"``, else ``None``.

    Kept out of the repr.
    """

    decay: float | None = None
    """The Nelson-Siegel decay the loadings were built at, else ``None``."""

    @property
    def k_factors(self) -> int:
        """Number of basis functions.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> FunctionalVAR(curves, grid, order=1, basis="bspline", df=6).fit().k_factors
            6
        """
        return int(self.basis_matrix.shape[1])

    @property
    def n_points(self) -> int:
        """Number of grid points per curve.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> FunctionalVAR(curves, grid, order=1).fit().n_points
            25
        """
        return int(self.basis_matrix.shape[0])

    @property
    def nobs(self) -> int:
        """Number of curves in the sample.

        The factor VAR's own ``nobs`` is smaller by its order, because the
        first ``order`` curves serve only as lags.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> res = FunctionalVAR(curves, grid, order=2).fit()
            >>> res.nobs, res.factors.nobs
            (80, 78)
        """
        return int(self.curves.shape[0])

    def projected_curves(self) -> npt.NDArray[np.float64]:
        """The sample as the basis sees it, over the full sample.

        Returns:
            An ``(nobs, n_points)`` array: each curve replaced by its
            projection onto the basis. The gap between this and
            :attr:`curves` is pure approximation error -- what the basis
            cannot represent -- before any dynamics enter.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
            >>> res = FunctionalVAR(curves, grid, order=1, basis="fpca", n_components=1).fit()
            >>> gap = curves - res.projected_curves()
            >>> gap.shape, bool(np.abs(gap).max() < 0.05)
            ((80, 25), True)
        """
        return self.mean_curve + self.scores @ self.basis_matrix.T

    @property
    def reconstruction_rmse(self) -> float:
        """Root-mean-square projection error over all curves and grid points.

        The floor for everything downstream: no forecast or fitted value can
        be closer to the data than the basis allows the data to be to itself.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
            >>> res = FunctionalVAR(curves, grid, order=1, basis="fpca", n_components=1).fit()
            >>> floor = res.reconstruction_rmse
            >>> one_step = np.sqrt(np.mean((curves[1:] - res.fitted_curves()) ** 2))
            >>> bool(floor < 0.02), bool(one_step > floor)
            (True, True)
        """
        return float(np.sqrt(np.mean((self.curves - self.projected_curves()) ** 2)))

    def fitted_curves(self) -> npt.NDArray[np.float64]:
        """One-step conditional mean curves over the factor VAR's effective sample.

        Returns:
            An ``(nobs - p, n_points)`` array, the factor VAR's fitted values
            mapped through the basis.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> res = FunctionalVAR(curves, grid, order=1).fit()
            >>> fitted = res.fitted_curves()
            >>> expected = res.mean_curve + res.factors.fittedvalues @ res.basis_matrix.T
            >>> fitted.shape, bool(np.allclose(fitted, expected))
            ((79, 25), True)
        """
        return self.mean_curve + self.factors.fittedvalues @ self.basis_matrix.T

    def forecast_curves(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Deterministic multi-step curve forecasts from the end of the sample.

        Args:
            steps: Forecast horizon, at least one.

        Returns:
            An array of shape ``(steps, n_points)``.

        Raises:
            SpecificationError: If ``steps`` is not positive, from the factor
                VAR's forecast.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> res = FunctionalVAR(curves, grid, order=1).fit()
            >>> path = res.forecast_curves(3)
            >>> path.shape, bool(np.abs(path[0] - curves[-1]).max() < 0.5)
            ((3, 25), True)
        """
        return self.mean_curve + self.factors.forecast(steps) @ self.basis_matrix.T

    def irf_curves(
        self, horizon: int = 20, *, orthogonalized: bool = True
    ) -> npt.NDArray[np.float64]:
        r"""The curve's response over time to a one-unit factor shock.

        The factor impulse response :math:`\Psi_h` of shape ``(r, r)`` is
        pushed through the basis as :math:`\Phi \Psi_h`, so a shock to
        factor ``j`` at lead ``h`` moves the curve by the ``j``-th column.
        For the Nelson-Siegel basis a level shock therefore shifts the whole
        curve by the same amount at lead zero without orthogonalization.

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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> res = FunctionalVAR(curves, grid, order=1, basis="fpca", n_components=1).fit()
            >>> response = res.irf_curves(4, orthogonalized=False)
            >>> response.shape, bool(np.allclose(response[0, :, 0], res.basis_matrix[:, 0]))
            ((5, 25, 1), True)
        """
        psi = self.factors.irf(horizon, orthogonalized=orthogonalized)
        return np.einsum("pr,hrj->hpj", self.basis_matrix, psi)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        One row per factor with its variance share (``"-"`` under a fixed
        basis) and mean absolute score; metadata naming the basis, the
        dimensions, the factor order and trend, the reconstruction floor and
        the factor log-likelihood; notes on factor stability, where
        factor-level inference lives, the Nelson-Siegel decay when one was
        used, and the reconstruction floor.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
            >>> res = FunctionalVAR(curves, grid, order=1, basis="bspline", df=4).fit()
            >>> table = res._summary_table()
            >>> table.title, table.columns
            ('FunctionalVAR(1) Results', ('factor', 'variance share', 'mean |score|'))
            >>> dict(table.metadata)["Basis"], [row[:2] for row in table.rows][:2]
            ('bspline', [('b1', '-'), ('b2', '-')])
        """
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
    r"""A vector autoregression for observations that are curves on a grid.

    Project, fit, map back. Each curve is reduced to its coordinates in a
    small basis, an ordinary VAR is estimated on those coordinates, and the
    curve-level surface -- fitted curves, forecast curves, curve impulse
    responses -- is the factor surface mapped through the basis. The choice of
    basis is the modelling decision, and the three on offer answer three
    different situations:

    ``"fpca"`` estimates the basis from the sample: the leading eigenfunctions
    of the empirical covariance, optimal in mean square for these data, with
    the component count chosen explicitly or by a variance-share target.

    ``"nelson-siegel"`` fixes the basis to level, slope, and curvature,

    .. math::

       \phi(\tau) = \Bigl(1,\;
       \frac{1 - e^{-\lambda\tau}}{\lambda\tau},\;
       \frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\Bigr),

    which is the Diebold-Li dynamic form of the term-structure literature. The
    grid is then a vector of maturities and must be strictly positive; the
    decay :math:`\lambda` is supplied, or profiled over the curvature-peak
    candidates the grid itself implies.

    ``"bspline"`` fixes a local polynomial basis with a chosen number of
    degrees of freedom -- the shape-agnostic option when neither the data-driven
    nor the yield-curve structure fits.

    Attributes:
        _curves: The validated ``(nobs, n_points)`` panel.
        _grid: The validated ``(n_points,)`` strictly increasing grid.
        _basis: The basis family.
        _order: Autoregressive order of the factor VAR.
        _trend: Deterministic terms of the factor VAR.
        _n_components: Retained components under ``"fpca"``, or ``None`` for
            the variance-share rule.
        _decay: The Nelson-Siegel decay, or ``None`` to profile it.
        _df: Spline degrees of freedom, resolved to ``min(8, n_points)``
            when not given.
        _degree: Spline degree.

    See Also:
        * :class:`FunctionalVARResult` -- the record ``fit()`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the factor model fitted on the scores.
        * :mod:`~cultivars.multivariate.reduced_form.term_structure` -- the
          dedicated yield-curve models when the Nelson-Siegel structure is
          the point rather than a basis.

    References:
        Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.

        Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data
        Analysis*. Springer.

    Example:
        >>> import numpy as np
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
        """Validate the curves, the grid, and the basis specification.

        Args:
            curves: The ``(nobs, n_points)`` panel, one curve per row.
            grid: The ``(n_points,)`` evaluation points. Defaults to
                ``0, 1, ...``; required in substance for ``"nelson-siegel"``,
                where it carries the maturities.
            order: Autoregressive order of the factor VAR.
            basis: One of ``"fpca"``, ``"nelson-siegel"``, ``"bspline"``.
            n_components: Retained components under ``"fpca"``. ``None`` keeps
                the smallest count explaining at least 95% of centered sample
                variance.
            decay: The Nelson-Siegel ``lambda``. ``None`` profiles it by
                reconstruction error over the candidates ``1.79 / tau`` for
                each grid maturity, the decay that puts the curvature peak at
                ``tau``.
            df: Spline degrees of freedom under ``"bspline"``; defaults to
                ``min(8, n_points)``.
            degree: Spline degree under ``"bspline"``.
            trend: Deterministic terms of the factor VAR.

        Raises:
            SpecificationError: If the basis, its parameters, or the grid are
                malformed for the chosen basis.
            DimensionError: If the shapes disagree. A sample too short for
                the factor VAR the projection implies is caught at ``fit()``.
            NumericalError: If the curves or the grid are non-finite.

        Example:
            >>> import numpy as np
            >>> curves = np.random.default_rng(0).standard_normal((40, 10))
            >>> model = FunctionalVAR(curves, order=1, basis="bspline", df=5)
            >>> model.basis, model.order
            ('bspline', 1)
            >>> FunctionalVAR(curves, np.arange(3), order=1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: grid must have one entry per curve column (10); ...
            >>> FunctionalVAR(curves, order=1, basis="nelson-siegel")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: nelson-siegel needs a strictly positive ...
            >>> FunctionalVAR(curves, order=1, basis="bspline", df=3)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: df must be an integer >= degree + 1 = 4; got 3.
        """
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
        """The basis family.

        Example:
            >>> import numpy as np
            >>> curves = np.random.default_rng(0).standard_normal((40, 10))
            >>> FunctionalVAR(curves, order=1).basis
            'fpca'
        """
        return self._basis

    @property
    def order(self) -> int:
        """Autoregressive order of the factor VAR.

        Example:
            >>> import numpy as np
            >>> curves = np.random.default_rng(0).standard_normal((40, 10))
            >>> FunctionalVAR(curves, order=2).order
            2
        """
        return self._order

    def _fpca_basis(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Estimate the principal-component basis from the centered sample.

        A thin SVD of the centered panel gives the eigenfunctions as right
        singular vectors and the variance shares as normalized squared
        singular values; the count is ``n_components`` or the smallest count
        whose cumulative share reaches 95%, capped by the numerical rank.

        Returns:
            The basis matrix, the mean curve, and the retained components'
            variance shares.

        Raises:
            SpecificationError: If ``n_components`` exceeds the sample's
                numerical rank.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
            >>> matrix, mean, shares = FunctionalVAR(curves, grid, order=1)._fpca_basis()
            >>> matrix.shape, bool(np.allclose(mean, curves.mean(axis=0))), bool(shares[0] > 0.95)
            ((25, 1), True, True)
            >>> too_many = FunctionalVAR(curves, grid, order=1, n_components=30)
            >>> too_many._fpca_basis()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_components (30) exceeds the sample's ...
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

        With a decay supplied the loadings are built once. Otherwise every
        grid maturity ``tau`` proposes the decay ``1.79 / tau`` that places
        the curvature loading's peak at ``tau``, and the candidate with the
        smallest projection error over the whole sample wins.

        Returns:
            The loading matrix and the decay it was built at.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
            >>> x = 0.6 * maturities
            >>> slope, curve = (1 - np.exp(-x)) / x, (1 - np.exp(-x)) / x - np.exp(-x)
            >>> factors = rng.standard_normal((60, 3)) * [0.2, 0.3, 0.4] + [5.0, -1.5, 0.5]
            >>> yields = factors @ np.column_stack([np.ones(8), slope, curve]).T
            >>> model = FunctionalVAR(yields, maturities, order=1, basis="nelson-siegel")
            >>> loadings, decay = model._nelson_siegel_basis()
            >>> loadings.shape, bool(abs(decay - 0.6) < 0.01)
            ((8, 3), True)
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
        """Build the clamped B-spline design on the grid.

        The ``df - degree - 1`` interior knots sit at equally spaced
        quantiles of the grid, and the boundary knots are repeated
        ``degree + 1`` times so the basis is clamped and sums to one at every
        grid point.

        Returns:
            The ``(n_points, df)`` design matrix.

        Example:
            >>> import numpy as np
            >>> curves = np.random.default_rng(0).standard_normal((40, 25))
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> design = FunctionalVAR(curves, grid, order=1, df=6)._bspline_basis()
            >>> design.shape, bool(np.allclose(design.sum(axis=1), 1.0))
            ((25, 6), True)
        """
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

        Builds the basis for the chosen family, projects the curves onto it
        -- centered by the sample mean under ``"fpca"``, uncentered under the
        fixed bases -- names the factors (``pc1 ...``, ``level``/``slope``/
        ``curvature``, or ``b1 ...``), and fits a
        :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
        of the requested order and trend to the scores.

        Returns:
            The fitted result, with the complete factor-level surface on its
            ``factors`` attribute.

        Raises:
            SpecificationError: If ``n_components`` exceeds the sample's
                numerical rank.
            DimensionError: If the sample is too short for the factor VAR,
                from the VAR's own validation.
            NumericalError: If a fixed basis is rank-deficient on the grid.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> grid = np.linspace(0.0, 1.0, 25)
            >>> curves = 0.1 * np.cumsum(rng.standard_normal(80))[:, None] + np.sin(np.pi * grid)
            >>> curves = curves + 0.01 * rng.standard_normal((80, 25))
            >>> res = FunctionalVAR(curves, grid, order=1, basis="bspline", df=5).fit()
            >>> res.basis, res.factors.names, res.explained_variance, res.decay
            ('bspline', ('b1', 'b2', 'b3', 'b4', 'b5'), None, None)
            >>> bool(np.allclose(res.mean_curve, 0.0)), bool(res.reconstruction_rmse < 0.02)
            (True, True)
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
