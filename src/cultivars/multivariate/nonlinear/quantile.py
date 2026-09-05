# filepath: /src/cultivars/multivariate/nonlinear/quantile.py
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

"""The quantile VAR: one autoregression per quantile level, no error term at all.

A quantile VAR models the *conditional quantiles* of each variable as linear
in the system's lags -- one full coefficient stack per requested level, so
the tails get their own dynamics rather than the mean's dynamics plus a
symmetric error. This is the natural language for macro-financial tail risk:
Adrian, Boyarchenko & Giannone's growth-at-risk finding is exactly that the
lower quantiles of GDP growth respond to financial conditions in a way the
median does not, which a conditional-mean VAR cannot express at any lag
length.

Estimation is the check-loss linear program of Koenker & Bassett, solved
exactly, equation by equation and level by level. Three consequences of
minimizing a check loss rather than a likelihood shape this surface, and each
is enforced rather than papered over. There is no ``llf``, no parameter-count
comparison, and no information criteria -- check loss is not a likelihood and
numbers shaped like those would rank nothing meaningful. There is no
orthogonalized impulse response or variance decomposition -- a quantile
system has no innovation covariance to factor, so :meth:`QVARResult.irf`
returns the reduced-form propagation of one level's coefficients and says
what linearization that is. And the forecast is one step only -- iterating a
quantile equation compounds quantiles of quantiles, which is not the
multi-step quantile of anything; the honest multi-horizon object is a direct
regression at each horizon, a different model the caller must state.

Separately fitted levels can also cross -- the estimated 90th percentile
path can dip below the estimated median in finite samples. The result
measures this (:attr:`QVARResult.crossing_share`) instead of silently
rearranging the fits.

References:
    Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*,
        46(1), 33-50.
    Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related
        inference processes for quantile regression. *Journal of the
        American Statistical Association*, 94(448), 1296-1310.
    White, H., Kim, T.-H., & Manganelli, S. (2015). VAR for VaR: Measuring
        tail dependence using multivariate regression quantiles. *Journal of
        Econometrics*, 187(1), 169-188.
    Adrian, T., Boyarchenko, N., & Giannone, D. (2019). Vulnerable growth.
        *American Economic Review*, 109(4), 1263-1289.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, companion_matrix
from ..._internals import (
    _QuantileVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorQuantileFit,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class QVARResult(_SummaryMixin):
    """A fitted quantile vector autoregression, all requested levels together.

    Deliberately absent: ``llf``, information criteria, orthogonalized
    responses, and multi-step forecasts. The module docstring states why
    each refusal is structural rather than an omission.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        quantiles: The estimated levels, ascending.
        coefficient_stacks: ``(Q, p, k, k)`` lag stacks, one per level.
        deterministics: ``(Q, n_det, k)`` deterministic coefficients.
        fittedvalues: ``(Q, nobs, k)`` in-sample conditional quantile paths.
        resid: ``(Q, nobs, k)`` quantile residuals ``y - fitted``.
        loss: ``(Q, k)`` total check loss per equation at the optimum.
        loss_location: ``(Q, k)`` check loss of the unconditional quantile,
            the benchmark :meth:`pseudo_r_squared` is read against.
        nobs: Effective sample size.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    quantiles: tuple[float, ...]
    coefficient_stacks: npt.NDArray[np.float64] = field(repr=False)
    deterministics: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    loss: npt.NDArray[np.float64] = field(repr=False)
    loss_location: npt.NDArray[np.float64] = field(repr=False)
    nobs: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorQuantileFit,
        model: _QuantileVectorAutoRegressionModel[QVARResult],
    ) -> QVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            quantiles=fit.quantiles,
            coefficient_stacks=fit.coefficient_stacks,
            deterministics=fit.deterministics,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            loss=fit.loss,
            loss_location=fit.loss_location,
            nobs=fit.nobs,
        )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def n_quantiles(self) -> int:
        """Number of estimated levels."""
        return len(self.quantiles)

    def _index_of(self, quantile: float) -> int:
        """Resolve a level to its position, refusing levels never estimated."""
        for i, tau in enumerate(self.quantiles):
            if abs(tau - quantile) <= 1e-9:
                return i
        raise SpecificationError(
            f"quantile {quantile} was not estimated; available levels are "
            f"{self.quantiles}. A new level is a new fit, not a lookup."
        )

    def coefficients(self, quantile: float) -> npt.NDArray[np.float64]:
        """One level's lag stack ``A_1(tau), ..., A_p(tau)``.

        Args:
            quantile: An estimated level.

        Returns:
            A ``(p, k, k)`` stack.

        Raises:
            SpecificationError: If the level was not estimated.
        """
        return self.coefficient_stacks[self._index_of(quantile)]

    def deterministic(self, quantile: float) -> npt.NDArray[np.float64]:
        """One level's deterministic coefficients, design-major.

        Args:
            quantile: An estimated level.

        Returns:
            An ``(n_det, k)`` block -- column ``i`` belongs to equation
            ``i``; empty when ``trend`` is ``"n"``.

        Raises:
            SpecificationError: If the level was not estimated.
        """
        return self.deterministics[self._index_of(quantile)]

    def quantile_path(self, name: str, quantile: float) -> npt.NDArray[np.float64]:
        """One variable's in-sample conditional quantile path.

        Args:
            name: An endogenous variable.
            quantile: An estimated level.

        Returns:
            An array of length ``nobs``.

        Raises:
            SpecificationError: If the variable or level is unknown.
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        return self.fittedvalues[self._index_of(quantile), :, self.names.index(name)]

    def pseudo_r_squared(self, quantile: float) -> dict[str, float]:
        """Koenker-Machado goodness of fit per equation at one level.

        ``1 - loss / loss_location``: the share of check loss the regression
        removes relative to the unconditional quantile. Local to its level
        by construction -- an ``R1`` at the median says nothing about the
        tails -- and not comparable to a least-squares ``R^2``.

        Args:
            quantile: An estimated level.

        Returns:
            A mapping from equation name to its ``R1``.

        Raises:
            SpecificationError: If the level was not estimated.
        """
        qi = self._index_of(quantile)
        ratio = self.loss[qi] / self.loss_location[qi]
        return {name: float(1.0 - ratio[i]) for i, name in enumerate(self.names)}

    @property
    def crossing_share(self) -> float:
        """Share of fitted points where adjacent quantile paths cross.

        Separately estimated levels are not constrained to be monotone, so
        the fitted 90th percentile can dip below the fitted median at some
        dates. This is the fraction of ``(date, variable, adjacent pair)``
        triples in violation -- a specification diagnostic, not an error:
        substantial crossing says the linear-in-lags quantile model is
        misspecified somewhere in the distribution. Zero when only one
        level was estimated.
        """
        if self.n_quantiles < 2:
            return 0.0
        return float(np.mean(np.diff(self.fittedvalues, axis=0) < 0.0))

    def ma_representation(self, horizon: int = 20, *, quantile: float) -> npt.NDArray[np.float64]:
        """Moving-average matrices of one level's coefficient stack.

        Args:
            horizon: Largest lead to return.
            quantile: An estimated level.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative or the level was
                not estimated.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        stack = self.coefficients(quantile)
        k, p = self.k_endog, self.order
        out = np.empty((horizon + 1, k, k))
        if p == 0:
            out[:] = 0.0
            out[0] = np.eye(k)
            return out
        selector = np.zeros((k, k * p))
        selector[:, :k] = np.eye(k)
        power = np.eye(k * p)
        companion = companion_matrix(stack)
        for h in range(horizon + 1):
            out[h] = selector @ power @ selector.T
            power = power @ companion
        return out

    def irf(
        self, horizon: int = 20, *, quantile: float, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Reduced-form propagation of one level's coefficients.

        Read this as "how a unit perturbation propagates if the tau-th
        quantile system were the law of motion" -- the pseudo-linearization
        the quantile-connectedness literature works with, stated as such.
        There is deliberately no ``orthogonalized`` option: quantile
        regression produces no innovation covariance, so there is nothing
        to factor and no recursive scheme to declare. Structural language
        on top of a quantile system is a modelling assumption this result
        will not smuggle in through a keyword.

        Args:
            horizon: Largest lead to return.
            quantile: An estimated level.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the response of variable ``i`` at lead ``h`` to a unit
            perturbation of equation ``j``.
        """
        out = self.ma_representation(horizon, quantile=quantile)
        return np.cumsum(out, axis=0) if cumulative else out

    def stability_profile(self) -> npt.NDArray[np.float64]:
        """Largest companion-root modulus of each level's system, ascending in tau.

        Tail systems are routinely more persistent than the median system --
        that asymmetry is often the finding -- and a modulus at or above one
        in a tail flags a level whose propagation reads explosively.

        Returns:
            An array of length ``Q``, aligned with :attr:`quantiles`.
        """
        if self.order == 0:
            return np.zeros(self.n_quantiles)
        out = np.empty(self.n_quantiles)
        for qi in range(self.n_quantiles):
            eigs = np.linalg.eigvals(companion_matrix(self.coefficient_stacks[qi]))
            out[qi] = float(np.abs(eigs).max(initial=0.0))
        return out

    def forecast(self, quantile: float) -> npt.NDArray[np.float64]:
        """The one-step-ahead conditional quantile of every variable.

        One step only, and deliberately so: iterating this equation would
        feed a quantile into a quantile, and the tau-quantile of a variable
        whose regressors are themselves tau-quantiles is not the two-step
        tau-quantile of the process. The multi-horizon object that means
        what it says is a *direct* quantile regression of ``y_{t+h}`` on
        date-``t`` information -- a different specification per horizon,
        which the caller states by fitting it.

        Args:
            quantile: An estimated level.

        Returns:
            A ``(k,)`` array: the next period's tau-th conditional quantile.

        Raises:
            SpecificationError: If the level was not estimated.
        """
        qi = self._index_of(quantile)
        n = self.endog.shape[0]
        det = {"n": [], "c": [1.0], "ct": [1.0, float(n + 1)]}[self.trend]
        out = np.asarray(det, dtype=np.float64) @ self.deterministics[qi]
        for lag in range(self.order):
            out = out + self.coefficient_stacks[qi, lag] @ self.endog[n - 1 - lag]
        return np.asarray(out, dtype=np.float64)

    def _specification(self) -> str:
        """Short specification label for display."""
        return f"QVAR({self.order})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        low, high = self.quantiles[0], self.quantiles[-1]
        lo_i, hi_i = self._index_of(low), self._index_of(high)
        rows = tuple(
            (
                name,
                f"{self.coefficient_stacks[lo_i, 0, i, i]:.4f}" if self.order else "-",
                f"{self.coefficient_stacks[hi_i, 0, i, i]:.4f}" if self.order else "-",
                f"{self.pseudo_r_squared(low)[name]:.3f}",
            )
            for i, name in enumerate(self.names)
        )
        roots = self.stability_profile()
        notes = [
            "Each level is an exact check-loss linear program (Koenker-"
            "Bassett); no likelihood, parameter count, or information "
            "criteria exist for it, so none are reported.",
            f"Max companion-root modulus across levels: {float(roots.max()):.4f} "
            "(stability_profile() gives the per-level values; tail systems "
            "are often the persistent ones).",
            f"Adjacent fitted quantile paths cross at "
            f"{100.0 * self.crossing_share:.2f}% of points; substantial "
            "crossing is a misspecification diagnostic.",
            "Impulse responses are reduced-form propagations of one level's "
            "coefficients; a quantile system has no innovation covariance, "
            "so orthogonalization is not offered.",
            "Forecasts are one step ahead only; multi-horizon quantiles "
            "require direct regressions at each horizon.",
            "Standard errors are not yet available for this estimator.",
        ]
        return SummaryTable(
            title=f"{self._specification()} Results",
            metadata=(
                ("Model", self._specification()),
                ("Levels", ", ".join(f"{q:g}" for q in self.quantiles)),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
            ),
            columns=(
                "equation",
                f"own L1 (tau={low:g})",
                f"own L1 (tau={high:g})",
                f"R1 (tau={low:g})",
            ),
            rows=rows,
            notes=tuple(notes),
        )


class QVAR(_QuantileVectorAutoRegressionModel[QVARResult]):
    """Quantile vector autoregression, Koenker-Bassett equation by equation.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> res = QVAR(y, order=1).fit(quantiles=(0.25, 0.5, 0.75))
        >>> res.coefficients(0.5).shape
        (1, 2, 2)
        >>> res.quantile_path("y1", 0.75).shape
        (199,)
    """

    __slots__ = ()

    def fit(self, *, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> QVARResult:
        """Estimate every equation at every requested level.

        Args:
            quantiles: Distinct levels, each strictly inside ``(0, 1)``.
                The default brackets the distribution the way the
                growth-at-risk literature reads it: both tails and the
                median.

        Returns:
            The fitted :class:`QVARResult`.

        Raises:
            SpecificationError: If the levels are malformed.
            NumericalError: If a program does not solve to optimality.
        """
        return QVARResult._from_fit(self._fit_quantile(quantiles), self)
