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
r"""The sparse VAR: shrinkage by selection, and the network it draws.

Where the Bayesian family shrinks every coefficient a little, the
penalized family sets most of them to exactly zero -- and in a large
system the zeros are the finding: which variables' histories actually
enter which equations is a directed Granger network, read straight off
the support. The estimator is one problem on the standardized panel,

.. math::

   \hat A = \arg\min_A\ \tfrac{1}{2}\sum_t
   \bigl\lVert y_t - c - \textstyle\sum_{i=1}^{p} A_i y_{t-i} \bigr\rVert^2
   + \lambda\, P(A),

under five penalties. The lasso :math:`P(A) = \sum |a_{ij}^{(l)}|`
(Basu and Michailidis, 2015) is the baseline; the adaptive lasso
reweights it by a ridge pilot, :math:`\sum |a| / |\tilde a|`; SCAD and
MCP (Nicholson-Matteson-Bien territory) unbias the large coefficients
the lasso drags toward zero, solved here by local linear approximation
-- a short sequence of weighted lasso problems, the standard route to
their oracle behavior (Zou and Li, 2008); and the lag-group penalty
:math:`\sum_l \sqrt{k^2}\,\lVert A_l \rVert_F` zeroes entire lag
matrices, which is lag selection rather than entry selection. Each
problem is solved by FISTA, warm-started down a geometric path from the
level that zeroes everything. Deterministic terms are never penalized,
and everything is standardized internally so one penalty level means
the same thing in every equation.

Two commitments shape the surface. First, the penalty level, when
unstated, is chosen by rolling-origin one-step forecast cross-validation
with refitting at every origin -- genuine out-of-sample errors, the
scheme the sparse-VAR literature settled on -- and its two consequences
are stated rather than hidden: prediction-optimal levels keep more
variables than the true support (the lasso especially; the nonconvex
penalties are the remedy when the support itself is the object), and the
path and its errors are kept on the result so the choice can be read
rather than trusted. Second, inference after selection is a genuinely
unsolved problem at this generality, so no likelihood, criteria or
standard errors are reported -- the result refuses to decorate selected
coefficients with intervals that ignore the selection, and the summary
says why.

The result satisfies the closed-system contract, so a fitted sparse VAR
feeds the structural layer and
:class:`~cultivars.multivariate.large_dim.spillover.Spillover` directly
-- estimating connectedness among a hundred banks with a lasso VAR is
exactly the Demirer-Diebold-Liu-Yilmaz pipeline.

Layout. :class:`SparseVAR` validates the specification on
``_SparseVectorAutoRegressionModel`` in ``_internals``, whose
``_fit_sparse`` standardizes, builds the geometric path from the zeroing
level, solves each level through ``_solve_penalized`` -- the lasso and
group penalties directly, the adaptive lasso with ridge-pilot weights,
SCAD and MCP through ``_nonconvex_weights`` and a reweighted pass -- on
``_fista_penalized`` from ``_internals._solvers``, selects by rolling
origin, and packs a ``_VectorSparseFit``. :class:`SparseVARResult` takes
the summary from ``_SummaryMixin`` and computes ``ma_representation``
and ``granger_adjacency`` itself. The penalty names are
:data:`~cultivars.typing.Penalty`. The second-stage precision network is
:mod:`~cultivars.multivariate.large_dim.graphical`; the shrinkage
alternatives are :mod:`~cultivars.multivariate.large_dim.bayesian` and
:mod:`~cultivars.multivariate.large_dim.gibbs`.

References:
    Basu, S., & Michailidis, G. (2015). Regularized estimation in sparse
    high-dimensional time series models. *Annals of Statistics*, 43(4),
    1535-1567.

    Nicholson, W. B., Matteson, D. S., & Bien, J. (2017). VARX-L:
    Structured regularization for large vector autoregressions with
    exogenous variables. *International Journal of Forecasting*, 33(3),
    627-651.

    Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave
    penalized likelihood models. *Annals of Statistics*, 36(4),
    1509-1533.

    Demirer, M., Diebold, F. X., Liu, L., & Yilmaz, K. (2018). Estimating
    global bank network connectedness. *Journal of Applied
    Econometrics*, 33(1), 1-15.

Example:
    A six-variable VAR(1) with two cross links, under the lasso and MCP:
    both recover the support, MCP with far fewer survivors, and the
    result feeds the spillover table directly:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.spillover import Spillover
    >>> rng = np.random.default_rng(0)
    >>> k, n = 6, 300
    >>> first = 0.5 * np.eye(k)
    >>> first[1, 0], first[3, 2] = 0.3, -0.3
    >>> y = np.zeros((n, k))
    >>> for t in range(1, n):
    ...     y[t] = first @ y[t - 1] + rng.standard_normal(k)
    >>> lasso = SparseVAR(y, order=1).fit(penalty="lasso")
    >>> mcp = SparseVAR(y, order=1).fit(penalty="mcp")
    >>> truth = [(1, 0), (3, 2)]
    >>> for res in (lasso, mcp):
    ...     edges = res.granger_adjacency(threshold=0.15)
    ...     print([(i, j) for i in range(k) for j in range(k) if edges[i, j]] == truth)
    True
    True
    >>> bool(mcp.n_nonzero <= lasso.n_nonzero / 2)
    True
    >>> table = Spillover(mcp, horizon=10).compute()
    >>> table.names == mcp.names, bool(0.0 <= table.total <= 100.0)
    (True, True)
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
    r"""A fitted sparse VAR: the coefficients that survived, and the zeros.

    The minimizer of

    .. math::

       \tfrac{1}{2}\sum_t \bigl\lVert y_t - c - \textstyle\sum_{i=1}^{p}
       A_i y_{t-i} \bigr\rVert^2 + \lambda\, P(A_1, \ldots, A_p)

    on the standardized panel, mapped back to original units, with
    :math:`P` the lasso, adaptive-lasso, SCAD, MCP or lag-group penalty
    and :math:`\lambda` either stated or chosen by rolling-origin
    one-step forecast error along a geometric path. Satisfies the
    closed-system contract, so identification models and
    :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
    consume it directly; the support it selects is read as a directed
    Granger network by :meth:`granger_adjacency`.

    Deliberately absent: ``llf``, information criteria, and standard
    errors. A penalized objective has no likelihood, its complexity is the
    nonzero count, and post-selection inference at this generality is an
    open problem the result will not paper over with naïve intervals.

    Note:
        ``sigma_u`` is the residual scatter divided by the sample size
        less the average number of active regressors per equation --
        the penalized nonzeros plus the deterministic terms -- so it is
        the degrees-of-freedom-corrected analogue of the OLS estimate,
        not the raw ``resid.T @ resid / nobs``. ``lam`` is on the
        standardized scale and comparable across equations but not
        across penalty families, whose paths start from different
        zeroing levels. ``granger_adjacency`` reads the support at
        ``threshold=0.0`` by default, so every surviving coefficient is
        an edge; the forecast-optimal penalty keeps many small ones, and
        a positive threshold or a nonconvex penalty gives the sparser
        network when support recovery is the object.

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

    See Also:
        * :class:`SparseVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.graphical.GraphicalVARResult`
          -- this record plus the sparse residual precision.
        * :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult`
          -- shrinkage by prior rather than selection, with inclusion
          probabilities instead of a support.

    References:
        Basu, S., & Michailidis, G. (2015). Regularized estimation in
        sparse high-dimensional time series models. *Annals of
        Statistics*, 43(4), 1535-1567.

        Nicholson, W. B., Matteson, D. S., & Bien, J. (2017). VARX-L:
        Structured regularization for large vector autoregressions with
        exogenous variables. *International Journal of Forecasting*,
        33(3), 627-651.

    Example:
        A six-variable VAR(2) with two cross-variable links; the lasso
        support recovers them and the moving-average matrices follow the
        recursion:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> k, n = 6, 300
        >>> first = 0.5 * np.eye(k)
        >>> first[1, 0], first[3, 2] = 0.3, -0.3
        >>> second = np.zeros((k, k))
        >>> second[0, 0] = 0.2
        >>> y = np.zeros((n, k))
        >>> for t in range(2, n):
        ...     y[t] = first @ y[t - 1] + second @ y[t - 2] + rng.standard_normal(k)
        >>> res = SparseVAR(y, order=2).fit(penalty="lasso")
        >>> res.penalty, res.k_endog, res.nobs, res.coefficients.shape
        ('lasso', 6, 298, (2, 6, 6))
        >>> res.lambda_path.shape, res.cv_errors.shape, bool(res.sparsity > 0.5)
        ((20,), (20,), True)
        >>> edges = res.granger_adjacency(threshold=0.15)
        >>> [(i, j) for i in range(k) for j in range(k) if edges[i, j]]
        [(1, 0), (3, 2)]
        >>> psi = res.ma_representation(3)
        >>> bool(np.allclose(psi[2], res.coefficients[0] @ psi[1] + res.coefficients[1]))
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed panel, shape ``(nobs_total, k)``. Kept out of the repr."""
    names: tuple[str, ...]
    """Variable labels, in column order."""
    order: int
    """The autoregressive order ``p``."""
    trend: str
    """The deterministic specification: ``"n"``, ``"c"`` or ``"ct"``."""
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    """``(p, k, k)`` lag stack at the selected penalty, original units. Kept out of the repr."""
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    """``(n_det, k)`` unpenalized deterministic coefficients. Kept out of the repr."""
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` degrees-of-freedom-corrected residual covariance. Kept out of the repr."""
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` residuals over the effective sample. Kept out of the repr."""
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` one-step means over the effective sample. Kept out of the repr."""
    penalty: str
    """The penalty family: ``lasso``, ``adaptive``, ``scad``, ``mcp`` or ``group``."""
    lam: float
    """The penalty level estimated at, on the standardized scale."""
    lambda_path: npt.NDArray[np.float64] = field(repr=False)
    """``(n_lambdas,)`` descending candidate path; empty when ``lam`` was stated."""
    cv_errors: npt.NDArray[np.float64] = field(repr=False)
    """``(n_lambdas,)`` rolling one-step squared errors; empty when ``lam`` was stated."""
    n_nonzero: int
    """Nonzero penalized coefficients at the solution."""
    nobs: int
    """Effective sample size after the ``p`` presample rows."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorSparseFit,
        model: _SparseVectorAutoRegressionModel[SparseVARResult],
    ) -> SparseVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed estimate from ``_fit_sparse``.
            model: The specification the fit was produced for.

        Returns:
            The public result record.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> model = SparseVAR(y, order=1)
            >>> fit = model._fit_sparse(
            ...     penalty="lasso", lam=0.1, n_lambdas=20, lambda_min_ratio=1e-3
            ... )
            >>> res = SparseVARResult._from_fit(fit, model)
            >>> res.names, res.lam, res.lambda_path.shape
            (('y1', 'y2', 'y3'), 0.1, (0,))
        """
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
        """Number of endogenous variables.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 4))
            >>> SparseVAR(y, order=1).fit(lam=0.1).k_endog
            4
        """
        return len(self.names)

    @property
    def sparsity(self) -> float:
        """Share of penalized coefficients set exactly to zero.

        ``1 - n_nonzero / (p k^2)``; the deterministic block is not
        penalized and not counted.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 4))
            >>> res = SparseVAR(y, order=1).fit(lam=0.5)
            >>> res.n_nonzero, res.sparsity
            (0, 1.0)
            >>> loose = SparseVAR(y, order=1).fit(lam=0.01)
            >>> bool(abs(loose.sparsity - (1 - loose.n_nonzero / 16)) < 1e-12)
            True
        """
        total = self.order * self.k_endog**2
        return 1.0 - self.n_nonzero / total if total else 0.0

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        r"""Moving-average matrices of the fitted system.

        :math:`\Psi_0 = I` and :math:`\Psi_h = \sum_{i=1}^{\min(h, p)}
        A_i \Psi_{h-i}`, the response of :math:`y_{t+h}` to a unit
        reduced-form innovation at :math:`t`.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 3))
            >>> for t in range(1, 200):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(3)
            >>> res = SparseVAR(y, order=1).fit(penalty="lasso")
            >>> psi = res.ma_representation(4)
            >>> psi.shape, bool(np.array_equal(psi[0], np.eye(3)))
            ((5, 3, 3), True)
            >>> bool(np.allclose(psi[3], np.linalg.matrix_power(res.coefficients[0], 3)))
            True
            >>> res.ma_representation(-1)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be non-negative; got -1.
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> k, n = 6, 300
            >>> first = 0.5 * np.eye(k)
            >>> first[1, 0], first[3, 2] = 0.3, -0.3
            >>> y = np.zeros((n, k))
            >>> for t in range(1, n):
            ...     y[t] = first @ y[t - 1] + rng.standard_normal(k)
            >>> res = SparseVAR(y, order=1).fit(penalty="mcp")
            >>> edges = res.granger_adjacency(threshold=0.15)
            >>> edges.dtype, bool(edges.diagonal().any())
            (dtype('bool'), False)
            >>> [(i, j) for i in range(k) for j in range(k) if edges[i, j]]
            [(1, 0), (3, 2)]
            >>> int(res.granger_adjacency().sum()) >= int(edges.sum())
            True
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
        """Build the structured summary.

        Returns:
            One row per equation with its count of nonzero lag regressors
            and the variable's Granger out-degree at threshold 0; the
            metadata carries the penalty, the level, the nonzero count
            and the sparsity; the notes state how the level was chosen,
            the forecast-versus-support caveat, the absence of inference,
            and the closed-system contract.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 3))
            >>> for t in range(1, 200):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(3)
            >>> table = SparseVAR(y, order=1).fit(penalty="lasso")._summary_table()
            >>> table.title, table.columns, len(table.rows), len(table.notes)
            ('Sparse VAR(1) Results', ('equation', 'nonzero regressors', 'out-degree'), 3, 4)
            >>> table.metadata[0], table.notes[0][-48:]
            (('Model', 'lasso-VAR(1)'), 'cross-validation with refitting at every origin.')
            >>> SparseVAR(y, order=1).fit(lam=0.1)._summary_table().notes[0][-22:]
            ' stated by the caller.'
        """
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
    r"""Penalized VAR: lasso, adaptive lasso, SCAD, MCP, or lag-group.

    One estimator, five penalties on the standardized lag coefficients.
    The lasso :math:`\lambda \sum |a|` is the baseline; the adaptive
    lasso reweights each coefficient by a ridge pilot,
    :math:`\lambda \sum |a| / |\tilde a|`; SCAD and MCP are the
    nonconvex penalties that leave large coefficients unbiased, solved
    by local linear approximation as a short sequence of weighted lasso
    problems; and the lag-group penalty
    :math:`\lambda \sum_i \sqrt{k^2}\, \lVert A_i \rVert_F` zeroes whole
    lag matrices, so it selects lags rather than entries. Every problem
    is solved by FISTA, warm-started along a geometric path from the
    level that zeroes everything down to ``lambda_min_ratio`` of it, and
    the level is chosen by rolling-origin one-step forecast error with a
    refit at every origin. Deterministic terms are never penalized.

    Attributes:
        _endog: The validated ``(nobs_total, k)`` panel.
        _order: The autoregressive order.
        _trend: The deterministic specification.
        _names: One label per variable.

    Args:
        endog: The observed panel, shape ``(nobs, k)``; wide is the point.
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
        * :class:`SparseVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.large_dim.graphical.GraphicalVAR`
          -- this estimator plus a sparse residual precision.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          shrinkage alternative for wide systems, with a likelihood.

    References:
        Basu, S., & Michailidis, G. (2015). Regularized estimation in
        sparse high-dimensional time series models. *Annals of
        Statistics*, 43(4), 1535-1567.

        Nicholson, W. B., Matteson, D. S., & Bien, J. (2017). VARX-L:
        Structured regularization for large vector autoregressions with
        exogenous variables. *International Journal of Forecasting*,
        33(3), 627-651.

        Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave
        penalized likelihood models. *Annals of Statistics*, 36(4),
        1509-1533.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 6))
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(6)
        >>> res = SparseVAR(y, order=2).fit(penalty="lasso")
        >>> res.coefficients.shape
        (2, 6, 6)
        >>> bool(res.sparsity > 0.3)
        True
        >>> SparseVAR(y, order=2).fit(penalty="ridge")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: penalty must be one of ('lasso', 'adaptive', ...
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
            lam: Penalty level, positive, or ``None`` for rolling-origin
                selection.
            n_lambdas: Candidate levels on the geometric path, at least 2.
            lambda_min_ratio: Smallest candidate as a fraction of the level
                that zeroes everything, in ``(0, 1)``.

        Returns:
            The fitted :class:`SparseVARResult`.

        Raises:
            SpecificationError: If the penalty specification is malformed
                or the sample cannot support the rolling validation.
            NumericalError: If the solver loses curvature.

        Example:
            The nonconvex penalties keep fewer coefficients than the lasso
            at their selected levels, and the group penalty keeps or drops
            whole lags:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> k, n = 6, 300
            >>> first = 0.5 * np.eye(k)
            >>> first[1, 0], first[3, 2] = 0.3, -0.3
            >>> y = np.zeros((n, k))
            >>> for t in range(1, n):
            ...     y[t] = first @ y[t - 1] + rng.standard_normal(k)
            >>> model = SparseVAR(y, order=2)
            >>> lasso = model.fit(penalty="lasso")
            >>> mcp = model.fit(penalty="mcp")
            >>> bool(mcp.n_nonzero < lasso.n_nonzero)
            True
            >>> grouped = model.fit(penalty="group")
            >>> per_lag = [int(np.count_nonzero(block)) for block in grouped.coefficients]
            >>> all(count in (0, k * k) for count in per_lag)
            True
            >>> model.fit(lam=0.0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: lam must be positive when given; got 0.0.
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
