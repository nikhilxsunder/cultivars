# filepath: /src/cultivars/multivariate/large_dim/sparse.py
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

"""The sparse VAR: shrinkage by selection, and the network it draws.

Where the Bayesian family shrinks every coefficient a little, the penalized
family sets most of them to exactly zero -- and in a large system the zeros
are the finding: which variables' histories actually enter which equations
is a directed Granger network, read straight off the support. Five penalty
families share one estimator. The lasso (Basu-Michailidis) is the baseline;
the adaptive lasso reweights it by a ridge pilot; SCAD and MCP
(Nicholson-Matteson-Bien territory) unbias the large coefficients the lasso
drags toward zero, solved here by local linear approximation -- a short
sequence of weighted lasso problems, the standard route to their oracle
behavior; and the lag-group penalty zeroes entire lag matrices, which is lag
selection rather than entry selection. Deterministic terms are never
penalized, and everything is standardized internally so one penalty level
means the same thing in every equation.

The penalty level, when unstated, is chosen by rolling-origin one-step
forecast cross-validation with refitting at every origin -- genuine
out-of-sample errors, the scheme the sparse-VAR literature settled on. Two
consequences are stated rather than hidden. Prediction-optimal levels keep
more variables than the true support (the lasso especially; the nonconvex
penalties are the remedy when the support itself is the object). And
inference after selection is a genuinely unsolved problem at this
generality, so no standard errors are reported -- the result refuses to
decorate selected coefficients with intervals that ignore the selection.

The result satisfies the closed-system contract, so a fitted sparse VAR
feeds the structural layer and
:class:`~cultivars.multivariate.large_dim.Spillover` directly -- estimating
connectedness among a hundred banks with a lasso VAR is exactly the
Demirer-Diebold-Liu-Yilmaz pipeline.

References:
    Basu, S., & Michailidis, G. (2015). Regularized estimation in sparse
        high-dimensional time series models. *Annals of Statistics*, 43(4),
        1535-1567.
    Nicholson, W. B., Matteson, D. S., & Bien, J. (2017). VARX-L: Structured
        regularization for large vector autoregressions with exogenous
        variables. *International Journal of Forecasting*, 33(3), 627-651.
    Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave
        penalized likelihood models. *Annals of Statistics*, 36(4),
        1509-1533.
    Demirer, M., Diebold, F. X., Liu, L., & Yilmaz, K. (2018). Estimating
        global bank network connectedness. *Journal of Applied
        Econometrics*, 33(1), 1-15.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import Penalty, SummaryTable
from ..._internals import (
    _SparseVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorSparseFit,
)
from ...exceptions import SpecificationError

__all__ = ["SparseVAR", "SparseVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SparseVARResult(_SummaryMixin):
    """A fitted sparse VAR: the coefficients that survived, and the zeros.

    Satisfies the closed-system contract, so identification models and
    :class:`~cultivars.multivariate.large_dim.Spillover` consume it
    directly.

    Deliberately absent: ``llf``, information criteria, and standard
    errors. A penalized objective has no likelihood, its complexity is the
    nonzero count, and post-selection inference at this generality is an
    open problem the result will not paper over with naïve intervals.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` lag stack at the selected penalty.
        deterministic: Unpenalized deterministic coefficients.
        sigma_u: Residual covariance, corrected by the average per-equation
            nonzero count.
        resid: Residuals over the effective sample.
        fittedvalues: One-step means over the effective sample.
        penalty: The penalty family estimated under.
        lam: The penalty level estimated at.
        lambda_path: The candidate path, descending; empty when the caller
            stated ``lam``.
        cv_errors: Rolling one-step squared forecast errors along the path;
            empty when the caller stated ``lam``.
        n_nonzero: Nonzero penalized coefficients at the solution.
        nobs: Effective sample size.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    penalty: str
    lam: float
    lambda_path: npt.NDArray[np.float64] = field(repr=False)
    cv_errors: npt.NDArray[np.float64] = field(repr=False)
    n_nonzero: int
    nobs: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorSparseFit,
        model: _SparseVectorAutoRegressionModel[SparseVARResult],
    ) -> SparseVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
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
        )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def sparsity(self) -> float:
        """Share of penalized coefficients set exactly to zero."""
        total = self.order * self.k_endog**2
        return 1.0 - self.n_nonzero / total if total else 0.0

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Moving-average matrices of the fitted system.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p = self.k_endog, self.order
        psi = np.empty((horizon + 1, k, k))
        psi[0] = np.eye(k)
        for h in range(1, horizon + 1):
            step = np.zeros((k, k))
            for lag in range(1, min(h, p) + 1):
                step += self.coefficients[lag - 1] @ psi[h - lag]
            psi[h] = step
        return psi

    def granger_adjacency(self, threshold: float = 0.0) -> npt.NDArray[np.bool_]:
        """The directed Granger network the support draws.

        Entry ``[i, j]`` is ``True`` when some lag of variable ``j`` enters
        variable ``i``'s equation with magnitude above ``threshold`` --
        ``j`` Granger-causes ``i`` in the selected model. A statement about
        the selected support, not a hypothesis test.

        Args:
            threshold: Magnitude below which an entry is read as absent.

        Returns:
            A ``(k, k)`` boolean matrix with a ``False`` diagonal.
        """
        strength = (
            np.abs(self.coefficients).max(axis=0)
            if self.order
            else np.zeros((self.k_endog, self.k_endog))
        )
        adjacency = strength > threshold
        np.fill_diagonal(adjacency, False)
        return adjacency

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        degree_out = self.granger_adjacency().sum(axis=0)
        rows = tuple(
            (
                name,
                f"{int(np.count_nonzero(self.coefficients[:, i, :]))}",
                f"{int(degree_out[i])}",
            )
            for i, name in enumerate(self.names)
        )
        notes = [
            f"Penalty {self.penalty} at lam={self.lam:.4g}"
            + (
                ", chosen by rolling-origin one-step forecast "
                "cross-validation with refitting at every origin."
                if self.lambda_path.size
                else ", stated by the caller."
            ),
            f"{self.n_nonzero} of {self.order * self.k_endog**2} penalized "
            f"coefficients survive ({100.0 * self.sparsity:.1f}% sparsity); "
            "prediction-optimal penalties keep more than the true support, "
            "and the nonconvex penalties (scad, mcp) are the remedy when "
            "support recovery is the object.",
            "No likelihood, information criteria, or standard errors are "
            "reported: post-selection inference at this generality is an "
            "open problem, and naïve intervals would ignore the "
            "selection.",
            "The result is a closed system: identification models and "
            "Spillover consume it directly.",
        ]
        return SummaryTable(
            title=f"Sparse VAR({self.order}) Results",
            metadata=(
                ("Model", f"{self.penalty}-VAR({self.order})"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("lam", f"{self.lam:.4g}"),
                ("Nonzero", f"{self.n_nonzero}"),
                ("Sparsity", f"{100.0 * self.sparsity:.1f}%"),
                ("Trend", self.trend),
            ),
            columns=("equation", "nonzero regressors", "out-degree"),
            rows=rows,
            notes=tuple(notes),
        )


class SparseVAR(_SparseVectorAutoRegressionModel[SparseVARResult]):
    """Penalized VAR: lasso, adaptive lasso, SCAD, MCP, or lag-group.

    Args:
        endog: The observed panel, shape ``(nobs, k)``; wide is the point.
        order: Autoregressive order.
        trend: Deterministic terms, never penalized.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 6))
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(6)
        >>> res = SparseVAR(y, order=2).fit(penalty="lasso")
        >>> res.coefficients.shape
        (2, 6, 6)
        >>> bool(res.sparsity > 0.3)
        True
    """

    __slots__ = ()

    def fit(
        self,
        *,
        penalty: Penalty = "lasso",
        lam: float | None = None,
        n_lambdas: int = 20,
        lambda_min_ratio: float = 1e-3,
    ) -> SparseVARResult:
        """Estimate under one penalty family, selecting the level if unstated.

        Args:
            penalty: ``"lasso"``, ``"adaptive"``, ``"scad"``, ``"mcp"``, or
                ``"group"`` (the lag-group penalty, which selects whole
                lags).
            lam: Penalty level, or ``None`` for rolling-origin selection.
            n_lambdas: Candidate levels on the geometric path.
            lambda_min_ratio: Smallest candidate as a fraction of the level
                that zeroes everything.

        Returns:
            The fitted :class:`SparseVARResult`.

        Raises:
            SpecificationError: If the penalty specification is malformed
                or the sample cannot support the rolling validation.
            NumericalError: If the solver loses curvature.
        """
        return SparseVARResult._from_fit(
            self._fit_sparse(
                penalty=penalty,
                lam=lam,
                n_lambdas=n_lambdas,
                lambda_min_ratio=lambda_min_ratio,
            ),
            self,
        )
