# filepath: /src/cultivars/multivariate/large_dim/graphical.py
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
r"""The graphical VAR: two networks, kept apart because they mean different things.

A large system carries two kinds of dependence and one covariance matrix
cannot show both. Dynamic dependence -- whose past moves whose present --
lives in the coefficient matrices, and its sparsity pattern is a directed
Granger network. Contemporaneous dependence -- who comoves with whom
within the period, given everyone else -- lives in the *precision* of
the innovations, and its sparsity pattern is an undirected
partial-correlation network. The graphical VAR (Barigozzi and
Brownlees, 2019) estimates both, separately and sparsely. The dynamics
are a lasso VAR,

.. math::

   \hat A = \arg\min_A\ \tfrac{1}{2}\sum_t
   \bigl\lVert y_t - c - \textstyle\sum_{i=1}^{p} A_i y_{t-i} \bigr\rVert^2
   + \lambda \sum_i \lVert A_i \rVert_1,

its penalty chosen by rolling-origin one-step forecast error along a
geometric path from the zeroing level; the precision is then estimated
on the residuals :math:`\hat u_t` by Meinshausen-Buhlmann nodewise
regressions, each residual on all the others,

.. math::

   \hat\gamma_{i\cdot} = \arg\min_\gamma\ \tfrac{1}{2}\sum_t
   \bigl(\hat u_{it} - \textstyle\sum_{j \ne i} \gamma_j \hat u_{jt}\bigr)^2
   + \lambda_i \lVert \gamma \rVert_1,
   \qquad
   \hat\Theta_{ij} = -\tfrac{1}{2}\bigl(\hat\theta_{ii}\hat\gamma_{ij}
   + \hat\theta_{jj}\hat\gamma_{ji}\bigr),

with per-node penalties :math:`\lambda_i` chosen by plain K-fold
cross-validation, which is legitimate in the second stage precisely
because residual rows carry no serial ordering worth respecting once the
dynamics are removed, and the two one-sided estimates symmetrized by
averaging.

Two commitments shape the surface. First, the two networks are read
with their own grammars and never merged. A Granger edge is directed and
temporal, a statement about the lasso support; a precision edge is
undirected and conditional -- zero means conditional independence given
the other innovations, exactly under Gaussianity and as a
partial-correlation zero otherwise. The result reports them through
separate methods with separate thresholds, and the summary counts them
separately. Second, neither network is a causal graph or a test. Edges
are statements about the selected model under the selected penalties;
no likelihood, information criteria or standard errors are reported,
and the summary says that post-selection inference is an open problem
rather than printing a p-value it cannot defend.

Layout. :class:`GraphicalVAR` validates the specification on
``_SparseVectorAutoRegressionModel`` in ``_internals`` and shares its
first stage with :class:`~cultivars.multivariate.large_dim.sparse.SparseVAR`:
``_fit_sparse`` builds the geometric path, solves each level with
``_fista_penalized`` from ``_internals._solvers`` warm-started along the
path, and selects by rolling-origin error; ``_fit_graphical`` then runs
the nodewise regressions on the residuals with the same solver under
K-fold selection and packs a ``_VectorGraphicalFit``.
:class:`GraphicalVARResult` extends
:class:`~cultivars.multivariate.large_dim.sparse.SparseVARResult`, which
supplies the closed-system surface, ``granger_adjacency`` and the path
diagnostics. The forecast-error-variance reading of connectedness is
:mod:`~cultivars.multivariate.large_dim.spillover`; the shrinkage
alternative to sparsity is :mod:`~cultivars.multivariate.large_dim.gibbs`.

References:
    Barigozzi, M., & Brownlees, C. (2019). NETS: Network estimation for
    time series. *Journal of Applied Econometrics*, 34(3), 347-364.

    Meinshausen, N., & Buhlmann, P. (2006). High-dimensional graphs and
    variable selection with the lasso. *Annals of Statistics*, 34(3),
    1436-1462.

    Basu, S., & Michailidis, G. (2015). Regularized estimation in sparse
    high-dimensional time series models. *Annals of Statistics*, 43(4),
    1535-1567.

Example:
    A chain in the dynamics and two pairs in the innovations; the two
    networks are recovered separately, and neither sees the other's
    structure:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> k, n = 5, 400
    >>> theta = np.eye(k)
    >>> theta[0, 1] = theta[1, 0] = -0.5
    >>> theta[2, 3] = theta[3, 2] = 0.5
    >>> chol = np.linalg.cholesky(np.linalg.inv(theta))
    >>> coef = 0.5 * np.eye(k)
    >>> coef[1, 0] = coef[2, 1] = 0.4
    >>> y = np.zeros((n, k))
    >>> for t in range(1, n):
    ...     y[t] = coef @ y[t - 1] + chol @ rng.standard_normal(k)
    >>> res = GraphicalVAR(y, order=1).fit(seed=0)
    >>> contemporaneous = res.contemporaneous_adjacency(threshold=0.3)
    >>> [(i, j) for i in range(k) for j in range(i + 1, k) if contemporaneous[i, j]]
    [(0, 1), (2, 3)]
    >>> granger = res.granger_adjacency(threshold=0.2)
    >>> [(i, j) for i in range(k) for j in range(k) if granger[i, j]]
    [(1, 0), (2, 1)]
    >>> bool(abs(res.coefficients[0][0, 1]) < 0.1), bool(abs(res.partial_correlations[1, 2]) < 0.1)
    (True, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable
from ..._internals import (
    _SparseVectorAutoRegressionModel,
    _VectorGraphicalFit,
)
from .sparse import SparseVARResult

__all__ = ["GraphicalVAR", "GraphicalVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GraphicalVARResult(SparseVARResult):
    r"""A fitted graphical VAR: the sparse dynamics plus the precision network.

    Everything the sparse result reports -- including the closed-system
    surface and :meth:`granger_adjacency` for the directed network -- plus
    the contemporaneous side: the residual precision
    :math:`\Theta = \Sigma_u^{-1}` estimated sparsely by nodewise lasso
    regressions of each residual on all the others,

    .. math::

       \hat\Theta_{ij} = -\tfrac{1}{2}\bigl(\hat\theta_{ii}\hat\gamma_{ij}
       + \hat\theta_{jj}\hat\gamma_{ji}\bigr),
       \qquad
       \rho_{ij} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}},

    with :math:`\hat\gamma_{ij}` the lasso slope of residual :math:`i` on
    residual :math:`j`, :math:`\hat\theta_{ii}` the inverse residual
    variance of that regression, the two one-sided estimates symmetrized
    by averaging, and :math:`\rho_{ij}` the partial correlation of
    innovations :math:`i` and :math:`j` given all the others.

    Note:
        The two networks have different grammars and different defaults.
        A Granger edge is directed and temporal, read off the lasso
        support at ``threshold=0.0`` by default, so every nonzero
        coefficient counts; the forecast-optimal penalty the rolling
        selection picks keeps many small coefficients, and the Granger
        network it draws is denser than the selection-optimal one.
        A contemporaneous edge is undirected and conditional, read off
        the partial correlations at ``threshold=0.05`` by default,
        because averaging a selected and an unselected one-sided slope
        leaves a small nonzero that is not an edge. Raise either
        threshold to read a sparser network; neither is a hypothesis
        test.

    Attributes:
        precision: ``(k, k)`` sparse residual precision from symmetrized
            nodewise regressions.
        partial_correlations: ``(k, k)`` partial correlations implied by
            the precision, unit diagonal.
        lam_nodes: ``(k,)`` per-node penalty levels chosen by K-fold
            cross-validation.

    See Also:
        * :class:`GraphicalVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.sparse.SparseVARResult`
          -- the dynamics-only record this one extends.
        * :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
          -- forecast-error-variance connectedness, the other network
          reading of a VAR.

    References:
        Barigozzi, M., & Brownlees, C. (2019). NETS: Network estimation
        for time series. *Journal of Applied Econometrics*, 34(3),
        347-364.

        Meinshausen, N., & Buhlmann, P. (2006). High-dimensional graphs
        and variable selection with the lasso. *Annals of Statistics*,
        34(3), 1436-1462.

    Example:
        Five innovations with two conditional dependencies, ``(1, 2)``
        and ``(3, 4)``; the precision network recovers them and the
        partial correlations follow from the precision:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> k, n = 5, 300
        >>> theta = np.eye(k)
        >>> theta[0, 1] = theta[1, 0] = -0.5
        >>> theta[2, 3] = theta[3, 2] = 0.5
        >>> chol = np.linalg.cholesky(np.linalg.inv(theta))
        >>> coef = 0.5 * np.eye(k)
        >>> coef[1, 0] = coef[2, 1] = 0.4
        >>> y = np.zeros((n, k))
        >>> for t in range(1, n):
        ...     y[t] = coef @ y[t - 1] + chol @ rng.standard_normal(k)
        >>> res = GraphicalVAR(y, order=1).fit(seed=0)
        >>> res.precision.shape, res.lam_nodes.shape, res.penalty
        ((5, 5), (5,), 'lasso')
        >>> bool(np.allclose(res.precision, res.precision.T))
        True
        >>> edges = res.contemporaneous_adjacency(threshold=0.3)
        >>> [(i, j) for i in range(k) for j in range(i + 1, k) if edges[i, j]]
        [(0, 1), (2, 3)]
        >>> bool(res.partial_correlations[0, 1] > 0.4), bool(res.partial_correlations[2, 3] < -0.4)
        (True, True)
    """

    precision: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` symmetrized nodewise-lasso residual precision. Kept out of the repr."""
    partial_correlations: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` partial correlations implied by ``precision``, unit diagonal. Kept out."""
    lam_nodes: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` per-node nodewise penalty levels from K-fold cross-validation. Kept out."""

    def contemporaneous_adjacency(self, threshold: float = 0.05) -> npt.NDArray[np.bool_]:
        """The undirected partial-correlation network.

        Entry ``[i, j]`` is ``True`` when the partial correlation between
        innovations ``i`` and ``j`` -- given all the others -- exceeds
        ``threshold`` in magnitude. Symmetric by construction.

        Args:
            threshold: Magnitude below which an edge is read as absent.

        Returns:
            A ``(k, k)`` boolean matrix with a ``False`` diagonal.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> k, n = 5, 300
            >>> theta = np.eye(k)
            >>> theta[0, 1] = theta[1, 0] = -0.5
            >>> theta[2, 3] = theta[3, 2] = 0.5
            >>> chol = np.linalg.cholesky(np.linalg.inv(theta))
            >>> y = np.zeros((n, k))
            >>> for t in range(1, n):
            ...     y[t] = 0.5 * y[t - 1] + chol @ rng.standard_normal(k)
            >>> res = GraphicalVAR(y, order=1).fit(seed=0)
            >>> edges = res.contemporaneous_adjacency()
            >>> edges.dtype, bool(np.array_equal(edges, edges.T)), bool(edges.diagonal().any())
            (dtype('bool'), True, False)
            >>> bool(edges[0, 1]), bool(edges[2, 3])
            (True, True)
            >>> int(res.contemporaneous_adjacency(threshold=0.3).sum() // 2)
            2
        """
        adjacency = np.abs(self.partial_correlations) > threshold
        np.fill_diagonal(adjacency, False)
        return adjacency

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per variable with its Granger out-degree and
            in-degree (support at threshold 0) and its contemporaneous
            degree (partial correlations above 0.05); the metadata carries
            the dimensions and both edge counts; the notes state the two
            networks' sizes, how each was estimated, how an edge is to be
            read, and that no inference is reported.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 4))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(4)
            >>> table = GraphicalVAR(y, order=1).fit(seed=0)._summary_table()
            >>> table.title, table.columns[:2]
            ('Graphical VAR(1) Results', ('variable', 'out-degree'))
            >>> len(table.rows), len(table.notes), table.metadata[0]
            (4, 4, ('Model', 'Graphical VAR(1)'))
        """
        granger = self.granger_adjacency()
        contemporaneous = self.contemporaneous_adjacency()
        rows = tuple(
            (
                name,
                f"{int(granger[i].sum())}",
                f"{int(granger[:, i].sum())}",
                f"{int(contemporaneous[i].sum())}",
            )
            for i, name in enumerate(self.names)
        )
        possible = self.k_endog * (self.k_endog - 1)
        notes = [
            f"Granger network: {int(granger.sum())} of {possible} directed "
            f"edges; contemporaneous network: "
            f"{int(contemporaneous.sum() // 2)} of {possible // 2} "
            "undirected edges at |partial correlation| > 0.05.",
            "The two networks answer different questions and are estimated "
            "separately: coefficients by lasso, the precision by nodewise "
            "regressions on the residuals (Meinshausen-Buhlmann), "
            "symmetrized by averaging.",
            "A precision zero is conditional independence given the other "
            "innovations under Gaussianity, a partial-correlation zero "
            "otherwise; edges are statements about the selected model, not "
            "hypothesis tests.",
            "No likelihood, information criteria, or standard errors are "
            "reported; post-selection inference is an open problem.",
        ]
        return SummaryTable(
            title=f"Graphical VAR({self.order}) Results",
            metadata=(
                ("Model", f"Graphical VAR({self.order})"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Granger edges", f"{int(granger.sum())}"),
                ("Contemporaneous edges", f"{int(contemporaneous.sum() // 2)}"),
                ("Trend", self.trend),
            ),
            columns=("variable", "out-degree", "in-degree", "contemporaneous degree"),
            rows=rows,
            notes=tuple(notes),
        )


class GraphicalVAR(_SparseVectorAutoRegressionModel[GraphicalVARResult]):
    r"""Graphical VAR, Barigozzi-Brownlees: sparse dynamics, sparse precision.

    Two stages. The dynamics are a lasso VAR,

    .. math::

       \min_{A}\ \tfrac{1}{2}\sum_t \lVert y_t - c - \textstyle\sum_i A_i y_{t-i}
       \rVert^2 + \lambda \sum_i \lVert A_i \rVert_1,

    with :math:`\lambda` fixed by the caller or chosen by rolling-origin
    forecast error along a path; the contemporaneous network is then
    estimated on the residuals by Meinshausen-Buhlmann nodewise lasso
    regressions -- each residual on all the others -- with each node's
    penalty chosen by K-fold cross-validation, which is legitimate in the
    second stage because residual rows carry no serial ordering once the
    dynamics are removed. Deterministic terms are never penalized.

    Attributes:
        _endog: The validated ``(nobs_total, k)`` panel.
        _order: The autoregressive order.
        _trend: The deterministic specification.
        _names: One label per variable.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order, at least 1.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``, never
            penalized.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If ``order`` is below 1, ``trend`` is not one
            of the three names, or ``names`` does not have one entry per
            column.
        DimensionError: If the panel is not a time-down-the-rows matrix or
            too short for the order.

    See Also:
        * :class:`GraphicalVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.large_dim.sparse.SparseVAR` --
          the first stage on its own, with the elastic-net and group
          penalties.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the unpenalized model, for systems small enough to afford it.

    References:
        Barigozzi, M., & Brownlees, C. (2019). NETS: Network estimation
        for time series. *Journal of Applied Econometrics*, 34(3),
        347-364.

        Meinshausen, N., & Buhlmann, P. (2006). High-dimensional graphs
        and variable selection with the lasso. *Annals of Statistics*,
        34(3), 1436-1462.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 5))
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(5)
        >>> res = GraphicalVAR(y, order=1).fit(seed=0)
        >>> res.precision.shape
        (5, 5)
        >>> res.contemporaneous_adjacency().dtype
        dtype('bool')
        >>> bool(res.lambda_path.size) and bool(res.cv_errors.size)
        True
        >>> fixed = GraphicalVAR(y, order=1).fit(lam=0.05, seed=0)
        >>> fixed.lam, fixed.lambda_path.shape, fixed.lam_nodes.shape
        (0.05, (0,), (5,))
    """

    __slots__ = ()

    def fit(
        self,
        *,
        lam: float | None = None,
        n_lambdas: int = 15,
        lambda_min_ratio: float = 1e-3,
        n_folds: int = 5,
        seed: int | np.random.Generator | None = None,
    ) -> GraphicalVARResult:
        """Estimate both networks.

        Args:
            lam: Stage-one (coefficient) penalty level, positive, or
                ``None`` for rolling-origin selection; the nodewise stage
                always selects its own per-node levels by K-fold
                cross-validation.
            n_lambdas: Candidate levels for both stages' paths, at least 2.
            lambda_min_ratio: Path floor as a fraction of the zeroing
                level, in ``(0, 1)``.
            n_folds: Cross-validation folds for the nodewise stage, at
                least 2.
            seed: Seed or generator for the fold shuffle.

        Returns:
            The fitted :class:`GraphicalVARResult`.

        Raises:
            SpecificationError: If a stage's specification is malformed.
            NumericalError: If the solver loses curvature.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 4))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(4)
            >>> model = GraphicalVAR(y, order=1)
            >>> res = model.fit(n_lambdas=10, n_folds=4, seed=0)
            >>> res.lambda_path.shape, res.cv_errors.shape, res.lam_nodes.shape
            ((10,), (10,), (4,))
            >>> again = model.fit(n_lambdas=10, n_folds=4, seed=0)
            >>> bool(np.array_equal(res.precision, again.precision))
            True
            >>> model.fit(n_folds=1)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_folds must be at least 2; got 1.
        """
        fit = self._fit_graphical(
            lam=lam,
            n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio,
            n_folds=n_folds,
            seed=seed,
        )
        assert isinstance(fit, _VectorGraphicalFit)
        return GraphicalVARResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            coefficients=fit.coefficient_stack,
            deterministic=fit.deterministic,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            penalty=fit.penalty,
            lam=fit.lam,
            lambda_path=fit.lambda_path,
            cv_errors=fit.cv_errors,
            n_nonzero=fit.n_nonzero,
            nobs=fit.nobs,
            precision=fit.precision,
            partial_correlations=fit.partial_correlations,
            lam_nodes=fit.lam_nodes,
        )
