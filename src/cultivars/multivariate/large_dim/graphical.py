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

"""The graphical VAR: two networks, kept apart because they mean different things.

A large system carries two kinds of dependence and one covariance matrix
cannot show both. Dynamic dependence -- whose past moves whose present --
lives in the coefficient matrices, and its sparsity pattern is a directed
Granger network. Contemporaneous dependence -- who comoves with whom within
the period, given everyone else -- lives in the *precision* of the
innovations, and its sparsity pattern is an undirected partial-correlation
network. The graphical VAR (Barigozzi-Brownlees) estimates both, separately
and sparsely: a lasso VAR for the dynamics, then Meinshausen-Buhlmann
nodewise regressions on its residuals for the precision, each residual on
all the others with per-node cross-validated penalties. Plain K-fold
cross-validation is legitimate in the second stage precisely because
residual rows carry no serial ordering worth respecting once the dynamics
are removed.

Read the two networks with their own grammars. A Granger edge is directed
and temporal; a precision edge is undirected and conditional -- zero means
conditional independence given the other innovations (exactly, under
Gaussianity; as a partial-correlation zero otherwise). Neither is a causal
graph, and edges are statements about the selected model, not hypothesis
tests.

References:
    Barigozzi, M., & Brownlees, C. (2019). NETS: Network estimation for
        time series. *Journal of Applied Econometrics*, 34(3), 347-364.
    Meinshausen, N., & Buhlmann, P. (2006). High-dimensional graphs and
        variable selection with the lasso. *Annals of Statistics*, 34(3),
        1436-1462.
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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GraphicalVARResult(SparseVARResult):
    """A fitted graphical VAR: the sparse dynamics plus the precision network.

    Everything the sparse result reports -- including the closed-system
    surface and :meth:`granger_adjacency` for the directed network -- plus
    the contemporaneous side.

    Attributes:
        precision: ``(k, k)`` sparse residual precision from symmetrized
            nodewise regressions.
        partial_correlations: ``(k, k)`` partial correlations implied by
            the precision, unit diagonal.
        lam_nodes: ``(k,)`` per-node penalty levels chosen by K-fold
            cross-validation.
    """

    precision: npt.NDArray[np.float64] = field(repr=False)
    partial_correlations: npt.NDArray[np.float64] = field(repr=False)
    lam_nodes: npt.NDArray[np.float64] = field(repr=False)

    def contemporaneous_adjacency(self, threshold: float = 0.05) -> npt.NDArray[np.bool_]:
        """The undirected partial-correlation network.

        Entry ``[i, j]`` is ``True`` when the partial correlation between
        innovations ``i`` and ``j`` -- given all the others -- exceeds
        ``threshold`` in magnitude. Symmetric by construction.

        Args:
            threshold: Magnitude below which an edge is read as absent.

        Returns:
            A ``(k, k)`` boolean matrix with a ``False`` diagonal.
        """
        adjacency = np.abs(self.partial_correlations) > threshold
        np.fill_diagonal(adjacency, False)
        return adjacency

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
    """Graphical VAR, Barigozzi-Brownlees: sparse dynamics, sparse precision.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        trend: Deterministic terms, never penalized.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 5))
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(5)
        >>> res = GraphicalVAR(y, order=1).fit(seed=0)
        >>> res.precision.shape
        (5, 5)
        >>> res.contemporaneous_adjacency().dtype
        dtype('bool')
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
            lam: Stage-one (coefficient) penalty level, or ``None`` for
                rolling-origin selection; the nodewise stage always selects
                its own per-node levels by K-fold cross-validation.
            n_lambdas: Candidate levels for both stages' paths.
            lambda_min_ratio: Path floor as a fraction of the zeroing level.
            n_folds: Cross-validation folds for the nodewise stage.
            seed: Seed or generator for the fold shuffle.

        Returns:
            The fitted :class:`GraphicalVARResult`.

        Raises:
            SpecificationError: If a stage's specification is malformed.
            NumericalError: If the solver loses curvature.
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
