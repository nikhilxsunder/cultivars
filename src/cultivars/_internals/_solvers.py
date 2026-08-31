# filepath: /src/cultivars/_internals/_solvers.py
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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core import link_matrix
from ..exceptions import DimensionError, NumericalError
from ._covariances import _PosteriorCovariance
from ._levels import _ConditionalLevels
from ._objectives import _Objective
from ._priors import _Prior, _PriorContext


def _solve[P](objective: _Objective[P]) -> tuple[P, float]:
    """Minimize an objective from every starting point and keep the best.

    Args:
        objective: The surface to minimize. Its ``method`` and ``options``
            select the algorithm; its ``starts`` supply the initial points.

    Returns:
        A tuple ``(parameters, criterion)`` where ``criterion`` is the minimized
        value of :meth:`_Objective.__call__` at the winning start.

    Raises:
        NumericalError: If ``starts`` yields no points at all.
    """
    best_x: npt.NDArray[np.float64] | None = None
    best_f = np.inf
    for theta0 in objective.starts():
        result = minimize(objective, theta0, method=objective.method, options=objective.options)
        if float(result.fun) < best_f:
            best_f = float(result.fun)
            best_x = np.asarray(result.x, dtype=np.float64)
    if best_x is None:
        raise NumericalError("objective supplied no starting points.")
    return objective.unpack(best_x), best_f


def _maximize_likelihood[P](objective: _Objective[P]) -> tuple[P, float]:
    """Solve a negative-log-likelihood objective, returning the log-likelihood.

    A thin sign-flip over :func:`_solve`, kept as a separate name so that no
    call site has to remember which surfaces are negated and which are not.

    Args:
        objective: A surface whose criterion is a negative log-likelihood.

    Returns:
        A tuple ``(parameters, llf)`` with the maximized log-likelihood.
    """
    parameters, criterion = _solve(objective)
    return parameters, -criterion


def solve_global(
    units: Sequence[_ConditionalLevels],
    *,
    weights: npt.NDArray[np.float64],
    unit_of_column: Sequence[int],
    variable_of_column: Sequence[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Stack conditional unit equations into one closed global autoregression.

    Each unit satisfies ``A_i0 z_it = d_it + sum_l A_il z_{i,t-l} + u_it`` with
    ``A_i0 = [I, -Lambda_i0]`` and ``A_il = [Phi_il, Lambda_il]``. Substituting
    ``z_it = W_i x_t`` writes every unit against the same global vector, and
    stacking the results gives ``G_0 x_t = d_t + sum_l G_l x_{t-l} + u_t``,
    which is square because the units' own blocks partition the global vector
    exactly once.

    The inversion of ``G_0`` is the moment the system closes. Individually no
    unit has a law of motion for its foreign variables; jointly they do, because
    one unit's foreign aggregate is a weighted sum of other units' domestic
    variables and those units have equations. Nothing is estimated here -- the
    result is an algebraic rearrangement of coefficients that were already
    fitted, and it reproduces every unit equation exactly rather than
    approximately.

    Args:
        units: One record per unit, in global column order.
        weights: A validated cross-unit weight matrix.
        unit_of_column: Owning unit index for each global column.
        variable_of_column: Variable identity for each global column.

    Returns:
        The contemporaneous matrix ``G_0``, the ``(p, k, k)`` stack of global
        autoregressive matrices ``G_0^{-1} G_l``, and the global deterministic
        coefficients.

    Raises:
        DimensionError: If the units do not partition the global columns.
        NumericalError: If ``G_0`` is singular, which means the contemporaneous
            linkage has no solution -- typically a weight matrix that makes two
            units' foreign aggregates identical.
    """
    width = len(unit_of_column)
    owners = tuple(int(u) for u in unit_of_column)
    covered = sorted(c for unit in range(len(units)) for c in range(width) if owners[c] == unit)
    if covered != list(range(width)):
        raise DimensionError(
            "the units must partition the global columns exactly once; "
            f"unit_of_column covers {covered} of {list(range(width))}."
        )
    order = max(unit.order for unit in units)
    depth = max(order, max(unit.exog_lags.shape[0] for unit in units))
    contemporaneous: list[npt.NDArray[np.float64]] = []
    lagged: list[list[npt.NDArray[np.float64]]] = [[] for _ in range(depth)]
    drift_rows = max(unit.deterministic.shape[0] for unit in units)
    drift = np.zeros((drift_rows, width), dtype=np.float64)
    cursor = 0
    for index, unit in enumerate(units):
        selector = link_matrix(
            index,
            weights=weights,
            unit_of_column=unit_of_column,
            variable_of_column=variable_of_column,
        )
        size = unit.k_endog
        contemporaneous.append(np.hstack([np.eye(size, dtype=np.float64), -unit.impact]) @ selector)
        for lag in range(depth):
            own = unit.phi[lag] if lag < unit.order else np.zeros((size, size))
            foreign = (
                unit.exog_lags[lag]
                if lag < unit.exog_lags.shape[0]
                else np.zeros((size, unit.k_exog))
            )
            lagged[lag].append(np.hstack([own, foreign]) @ selector)
        rows = unit.deterministic.shape[0]
        drift[:rows, cursor : cursor + size] = unit.deterministic
        cursor += size
    g_zero = np.vstack(contemporaneous)
    if abs(float(np.linalg.slogdet(g_zero)[0])) < 0.5:
        raise NumericalError(
            "the contemporaneous linkage matrix is singular, so the global system has no "
            "solution. The usual cause is a weight matrix that gives two units identical "
            "foreign aggregates, which leaves their contemporaneous equations linearly "
            "dependent."
        )
    blocks = np.stack([np.linalg.solve(g_zero, np.vstack(lagged[lag])) for lag in range(depth)])
    return g_zero, blocks, np.linalg.solve(g_zero, drift.T).T


def posterior_coefficients(
    target: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    prior: _Prior,
    context: _PriorContext,
) -> _PosteriorCovariance:
    """Posterior mean of the coefficient matrix, equation by equation.

    Each equation solves ``(X'X / s_i^2 + V_i^-1) b_i = X'y_i / s_i^2 +
    V_i^-1 m_i`` with ``V_i`` the prior variances for that equation and ``m_i``
    its prior mean. Running equation by equation is not an approximation and
    not a shortcut: once the cross-equation weight differs from one, the prior
    variance stops factoring as a Kronecker product, the Normal-inverse-Wishart
    conjugacy is gone, and Litterman's original per-equation procedure is the
    correct one.

    Any dummy rows the prior contributes are stacked under the sample first,
    which is how a restriction on a *sum* of coefficients enters a problem
    whose other restrictions are diagonal variances.

    Args:
        target: The ``(nobs, k)`` block being explained.
        design: The ``(nobs, width)`` regressor matrix.
        prior: The prior, possibly a composition.
        context: What the prior needs to know about the sample.

    Returns:
        A :class:`_PosteriorCovariance` carrying the posterior mean, the
        per-equation covariance, and the effective parameter count. The last is
        not decoration: shrinkage means the model spends fewer parameters than
        the design has columns, and an information criterion that charges for
        the nominal width penalizes a shrunk model for freedom it never used.

    Raises:
        DimensionError: If the design does not match the context's width.
    """
    if design.shape[1] != context.width:
        raise DimensionError(
            f"design has {design.shape[1]} columns but the prior context describes "
            f"{context.width}. The two must agree on the column order or the prior "
            "lands on the wrong coefficients."
        )
    dummy_target, dummy_design = prior.dummy_observations(context)
    full_target = np.vstack([target, dummy_target]) if dummy_target.shape[0] else target
    full_design = np.vstack([design, dummy_design]) if dummy_design.shape[0] else design
    mean = prior.coefficient_mean(context)
    variance = prior.coefficient_variance(context)
    cross = full_design.T @ full_design
    moment = full_design.T @ full_target
    out = np.empty((context.width, context.k_endog), dtype=np.float64)
    blocks = np.empty((context.k_endog, context.width, context.width), dtype=np.float64)
    effective = 0.0
    for index in range(context.k_endog):
        scale = float(context.scales[index]) ** 2
        column = variance[:, index]
        precision = np.where(np.isfinite(column), 1.0 / np.maximum(column, 1e-300), 0.0)
        left = cross / scale + np.diag(precision)
        right = moment[:, index] / scale + precision * mean[:, index]
        out[:, index] = np.linalg.solve(left, right)
        posterior = np.linalg.inv(left)
        blocks[index] = posterior
        effective += float(np.trace(posterior @ cross)) / scale
    return _PosteriorCovariance(coefficients=out, blocks=blocks, effective_parameters=effective)
