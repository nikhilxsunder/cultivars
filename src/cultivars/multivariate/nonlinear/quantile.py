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
r"""The quantile VAR: one autoregression per quantile level, no error term at all.

A quantile VAR models the *conditional quantiles* of each variable as
linear in the system's lags,

.. math::

   Q_\tau(y_{i,t} \mid y_{t-1}, \ldots, y_{t-p})
   = c_i(\tau)' d_t + \sum_{l=1}^{p} a_{i,l}(\tau)' y_{t-l},
   \qquad \tau \in (0, 1),

one full coefficient stack per requested level, so the tails get their
own dynamics rather than the mean's dynamics plus a symmetric error. This
is the natural language for macro-financial tail risk: Adrian,
Boyarchenko and Giannone's growth-at-risk finding is exactly that the
lower quantiles of GDP growth respond to financial conditions in a way the
median does not, which a conditional-mean VAR cannot express at any lag
length. Estimation is the check-loss program of Koenker and Bassett,

.. math::

   \hat\beta_i(\tau) = \arg\min_\beta \sum_t \rho_\tau(y_{i,t} - x_t'\beta),
   \qquad \rho_\tau(e) = e\,(\tau - \mathbf{1}\{e < 0\}),

solved exactly as a linear program, equation by equation and level by
level, since the loss is additive across both.

Two commitments shape the surface. First, three consequences of
minimizing a check loss rather than a likelihood are enforced rather than
papered over. There is no ``llf``, no parameter-count comparison, and no
information criteria -- check loss is not a likelihood and numbers shaped
like those would rank nothing meaningful; the Koenker-Machado
:math:`R^1`, local to its level, is the goodness of fit that does exist
and is reported. There is no orthogonalized impulse response or variance
decomposition -- a quantile system has no innovation covariance to
factor, so :meth:`QVARResult.irf` returns the reduced-form propagation of
one level's coefficients and says what linearization that is; for the
same reason the result does not satisfy the closed-system contract and
cannot be passed to the structural layer or to the spillover view. And
the forecast is one step only -- iterating a quantile equation compounds
quantiles of quantiles, which is not the multi-step quantile of anything;
the honest multi-horizon object is a direct regression at each horizon,
a different model the caller must state. Second, separately fitted levels
can cross -- the estimated 90th percentile path can dip below the
estimated median in finite samples -- and the result measures this
(:attr:`QVARResult.crossing_share`) instead of silently rearranging the
fits. Standard errors are not yet offered; the summary says so.

Layout. :class:`QVAR` validates on
``_QuantileVectorAutoRegressionModel`` in ``_internals``, which
re-specifies the linear VAR base without its prior surface and holds the
engine: ``_quantile_regression`` solves one equation at one level through
``scipy.optimize.linprog`` with the HiGHS backend on a sparse constraint
matrix, ``_check_loss`` scores a residual block, and ``_fit_quantile``
validates and sorts the levels, runs the :math:`k \times Q` programs over
the shared design, computes the unconditional-quantile benchmark losses,
and packs a ``_VectorQuantileFit``. :class:`QVARResult` takes its summary
from ``_SummaryMixin`` and builds each level's companion through
:func:`~cultivars._core.companion_matrix`. The conditional-mean system on
the same design is
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`; the
other members of this package model the mean's coefficients as functions
of a state or of time.

References:
    Koenker, R., & Bassett, G. (1978). Regression quantiles.
    *Econometrica*, 46(1), 33-50.

    Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related
    inference processes for quantile regression. *Journal of the American
    Statistical Association*, 94(448), 1296-1310.

    White, H., Kim, T.-H., & Manganelli, S. (2015). VAR for VaR: Measuring
    tail dependence using multivariate regression quantiles. *Journal of
    Econometrics*, 187(1), 169-188.

    Adrian, T., Boyarchenko, N., & Giannone, D. (2019). Vulnerable growth.
    *American Economic Review*, 109(4), 1263-1289.

Example:
    Growth at risk in two lines. Financial conditions scale the downside
    shocks to growth, so the lower-tail coefficient on lagged conditions
    is far larger than the median's while a conditional-mean VAR sees one
    number:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((400, 2))
    >>> for t in range(1, 400):
    ...     fci = 0.6 * y[t - 1, 1] + rng.standard_normal()
    ...     shock = rng.standard_normal()
    ...     if shock < 0.0:
    ...         shock *= np.exp(0.4 * y[t - 1, 1])
    ...     y[t] = [0.3 * y[t - 1, 0] - 0.2 * y[t - 1, 1] + shock, fci]
    >>> res = QVAR(y, order=1, names=("growth", "fci")).fit(quantiles=(0.1, 0.5, 0.9))
    >>> on_fci = {q: round(float(res.coefficients(q)[0, 0, 1]), 2) for q in res.quantiles}
    >>> bool(on_fci[0.1] < -0.5), bool(-0.3 < on_fci[0.5] < 0.0), bool(-0.3 < on_fci[0.9] < 0.0)
    (True, True, True)
    >>> mean = VAR(y, order=1).fit()
    >>> bool(-0.4 < mean.coefficients[0, 0, 1] < -0.1)
    True
    >>> bool(res.crossing_share < 0.01), bool(res.pseudo_r_squared(0.1)["growth"] > 0.1)
    (True, True)
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

__all__ = ["QVAR", "QVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class QVARResult(_SummaryMixin):
    r"""A fitted quantile vector autoregression, all requested levels together.

    For each level :math:`\tau` in :attr:`quantiles` and each equation
    :math:`i`, the coefficients solve the check-loss program

    .. math::

       \min_{c_i(\tau),\, a_i(\tau)} \sum_t
       \rho_\tau\!\Bigl(y_{i,t} - c_i(\tau)' d_t - \sum_{l=1}^{p}
       a_{i,l}(\tau)' y_{t-l}\Bigr),
       \qquad
       \rho_\tau(e) = e\,(\tau - \mathbf{1}\{e < 0\}),

    so :math:`Q_\tau(y_{i,t} \mid y_{t-1}, \ldots)` is linear in the
    system's lags with its own coefficient stack at every level. The
    programs separate across equations and across levels, and each is
    solved exactly; the stacks are collected level-major, ascending in
    :math:`\tau`.

    Deliberately absent: ``llf``, information criteria, orthogonalized
    responses, and multi-step forecasts. The module docstring states why
    each refusal is structural rather than an omission.

    Note:
        Every accessor that takes a ``quantile`` resolves it against the
        levels actually estimated (to within ``1e-9``) and refuses any
        other -- a new level is a new fit, not an interpolation. The fitted
        paths at a level are exact in-sample quantiles in the sense of the
        program: the share of observations below the fitted
        :math:`\tau`-path is :math:`\tau` up to the discreteness of the
        sample. Levels are fitted independently, so the paths may cross;
        :attr:`crossing_share` measures how often, and the result does not
        rearrange them. ``resid`` is ``y - fitted`` at each level, a
        quantile residual whose sign pattern, not its magnitude, carries
        the fit.

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

    See Also:
        * :class:`QVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the conditional-mean system whose single stack these levels
          fan out from.
        * :class:`~cultivars.univariate.conditional_variance.GARCHResult`
          -- the other account of a moving tail, through the variance.

    References:
        Koenker, R., & Bassett, G. (1978). Regression quantiles.
        *Econometrica*, 46(1), 33-50.

        Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related
        inference processes for quantile regression. *Journal of the
        American Statistical Association*, 94(448), 1296-1310.

        White, H., Kim, T.-H., & Manganelli, S. (2015). VAR for VaR:
        Measuring tail dependence using multivariate regression quantiles.
        *Journal of Econometrics*, 187(1), 169-188.

    Example:
        A growth-at-risk system: ``growth`` depends on lagged financial
        conditions ``fci`` at the median, and its *downside* depends on
        them more, because the negative shocks are scaled by ``fci``. The
        lower-tail coefficient is roughly three times the median one, the
        10th-percentile path sits below a tenth of the observations, and
        the paths barely cross:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((400, 2))
        >>> for t in range(1, 400):
        ...     fci = 0.6 * y[t - 1, 1] + rng.standard_normal()
        ...     shock = rng.standard_normal()
        ...     if shock < 0.0:
        ...         shock *= np.exp(0.4 * y[t - 1, 1])
        ...     y[t] = [0.3 * y[t - 1, 0] - 0.2 * y[t - 1, 1] + shock, fci]
        >>> res = QVAR(y, order=1, names=("growth", "fci")).fit(quantiles=(0.1, 0.5, 0.9))
        >>> res.quantiles, res.n_quantiles, res.k_endog, res.nobs
        ((0.1, 0.5, 0.9), 3, 2, 399)
        >>> res.coefficient_stacks.shape, res.fittedvalues.shape, res.loss.shape
        ((3, 1, 2, 2), (3, 399, 2), (3, 2))
        >>> tail, median = res.coefficients(0.1)[0, 0, 1], res.coefficients(0.5)[0, 0, 1]
        >>> bool(tail < 2.5 * median < 0.0)
        True
        >>> below = y[1:, 0] < res.quantile_path("growth", 0.1)
        >>> bool(abs(below.mean() - 0.1) < 0.01)
        True
        >>> bool(res.crossing_share < 0.01)
        True
        >>> bool(np.all(np.diff([res.forecast(q)[0] for q in res.quantiles]) > 0.0))
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
    quantiles: tuple[float, ...]
    """The estimated levels, ascending, each strictly inside ``(0, 1)``."""
    coefficient_stacks: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, p, k, k)`` lag stacks, level-major. Kept out of the repr."""
    deterministics: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, n_det, k)`` deterministic coefficients, level-major. Kept out of the repr."""
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, nobs, k)`` in-sample conditional quantile paths. Kept out of the repr."""
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, nobs, k)`` quantile residuals ``y - fitted``. Kept out of the repr."""
    loss: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, k)`` total check loss per equation at the optimum. Kept out of the repr."""
    loss_location: npt.NDArray[np.float64] = field(repr=False)
    """``(Q, k)`` check loss of the unconditional sample quantile. Kept out of the repr."""
    nobs: int
    """Effective sample size after the ``p`` presample rows."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorQuantileFit,
        model: _QuantileVectorAutoRegressionModel[QVARResult],
    ) -> QVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed level-by-level programs.
            model: The validated specification they were solved on.

        Returns:
            The frozen :class:`QVARResult`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = QVAR(y, order=1)
            >>> res = QVARResult._from_fit(model._fit_quantile((0.5, 0.25)), model)
            >>> res.quantiles, res.names
            ((0.25, 0.5), ('y1', 'y2'))
        """
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
        """Number of endogenous variables.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> QVAR(y, order=1).fit(quantiles=(0.5,)).k_endog
            3
        """
        return len(self.names)

    @property
    def n_quantiles(self) -> int:
        """Number of estimated levels.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> QVAR(y, order=1).fit(quantiles=(0.05, 0.5, 0.95)).n_quantiles
            3
        """
        return len(self.quantiles)

    def _index_of(self, quantile: float) -> int:
        """Resolve a level to its position, refusing levels never estimated.

        Args:
            quantile: A level, matched to within ``1e-9``.

        Returns:
            Its index along the level axis of every ``(Q, ...)`` array.

        Raises:
            SpecificationError: If no estimated level matches.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))
            >>> res._index_of(0.9), res._index_of(0.1 + 1e-12)
            (2, 0)
            >>> res._index_of(0.3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: quantile 0.3 was not estimated; available ...
        """
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
            A ``(p, k, k)`` stack; ``[l - 1, i, j]`` is the coefficient on
            lag ``l`` of variable ``j`` in the ``tau``-quantile equation of
            variable ``i``.

        Raises:
            SpecificationError: If the level was not estimated.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = QVAR(y, order=1).fit(quantiles=(0.25, 0.5, 0.75))
            >>> res.coefficients(0.5).shape
            (1, 2, 2)
            >>> bool(np.all(np.abs(np.diag(res.coefficients(0.5)[0]) - 0.5) < 0.15))
            True
            >>> bool(np.allclose(res.coefficients(0.75), res.coefficient_stacks[2]))
            True
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

        Example:
            On symmetric innovations the intercepts order with the level:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((300, 2))
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))
            >>> res.deterministic(0.5).shape
            (1, 2)
            >>> bool(np.all(res.deterministic(0.1) < res.deterministic(0.5)))
            True
            >>> bool(np.all(res.deterministic(0.5) < res.deterministic(0.9)))
            True
            >>> QVAR(y, order=1, trend="n").fit(quantiles=(0.5,)).deterministic(0.5).shape
            (0, 2)
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

        Example:
            The share of observations below the fitted ``tau``-path is
            ``tau``, the defining property of the check-loss optimum:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((300, 2))
            >>> res = QVAR(y, order=1).fit(quantiles=(0.25, 0.75))
            >>> path = res.quantile_path("y2", 0.25)
            >>> path.shape, bool(abs(np.mean(y[1:, 1] < path) - 0.25) < 0.01)
            ((299,), True)
            >>> res.quantile_path("y3", 0.25)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown variable 'y3'; expected one of ...
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        return self.fittedvalues[self._index_of(quantile), :, self.names.index(name)]

    def pseudo_r_squared(self, quantile: float) -> dict[str, float]:
        r"""Koenker-Machado goodness of fit per equation at one level.

        ``1 - loss / loss_location``: the share of check loss the regression
        removes relative to the unconditional quantile,

        .. math::

           R^1(\tau) = 1 - \frac{\sum_t \rho_\tau(y_t - \hat Q_\tau(y_t \mid x_t))}
           {\sum_t \rho_\tau(y_t - \hat q_\tau)}.

        Local to its level by construction -- an ``R1`` at the median says
        nothing about the tails -- and not comparable to a least-squares
        ``R^2``.

        Args:
            quantile: An estimated level.

        Returns:
            A mapping from equation name to its ``R1``.

        Raises:
            SpecificationError: If the level was not estimated.

        Example:
            A persistent system explains more check loss than white noise at
            every level, and every ``R1`` lies in ``[0, 1]``:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5))
            >>> fit = res.pseudo_r_squared(0.5)
            >>> sorted(fit), bool(all(0.0 <= v <= 1.0 for v in fit.values()))
            (['y1', 'y2'], True)
            >>> noise = QVAR(rng.standard_normal((300, 2)), order=1).fit(quantiles=(0.5,))
            >>> bool(fit["y1"] > noise.pseudo_r_squared(0.5)["y1"])
            True
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

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((300, 2))
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))
            >>> bool(0.0 <= res.crossing_share < 0.05)
            True
            >>> QVAR(y, order=1).fit(quantiles=(0.5,)).crossing_share
            0.0
        """
        if self.n_quantiles < 2:
            return 0.0
        return float(np.mean(np.diff(self.fittedvalues, axis=0) < 0.0))

    def ma_representation(self, horizon: int = 20, *, quantile: float) -> npt.NDArray[np.float64]:
        """Moving-average matrices of one level's coefficient stack.

        The companion powers of ``coefficients(quantile)`` read back into
        ``(k, k)`` blocks -- the linear propagation the level's system
        would have if it were a law of motion.

        Args:
            horizon: Largest lead to return.
            quantile: An estimated level.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative or the level was
                not estimated.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = QVAR(y, order=1).fit(quantiles=(0.5,))
            >>> psi = res.ma_representation(3, quantile=0.5)
            >>> psi.shape, bool(np.allclose(psi[0], np.eye(2)))
            ((4, 2, 2), True)
            >>> bool(np.allclose(psi[1], res.coefficients(0.5)[0]))
            True
            >>> bool(np.allclose(psi[2], psi[1] @ psi[1]))
            True
            >>> res.ma_representation(-1, quantile=0.5)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be non-negative; got -1.
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

        Raises:
            SpecificationError: If ``horizon`` is negative or the level was
                not estimated.

        Example:
            On the growth-at-risk system the lower-tail response of growth
            to a financial-conditions perturbation is larger in magnitude
            than the median response:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((400, 2))
            >>> for t in range(1, 400):
            ...     fci = 0.6 * y[t - 1, 1] + rng.standard_normal()
            ...     shock = rng.standard_normal()
            ...     if shock < 0.0:
            ...         shock *= np.exp(0.4 * y[t - 1, 1])
            ...     y[t] = [0.3 * y[t - 1, 0] - 0.2 * y[t - 1, 1] + shock, fci]
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5))
            >>> tail, median = res.irf(4, quantile=0.1), res.irf(4, quantile=0.5)
            >>> tail.shape, bool(np.allclose(tail[0], np.eye(2)))
            ((5, 2, 2), True)
            >>> bool(abs(tail[1, 0, 1]) > abs(median[1, 0, 1]))
            True
            >>> running = res.irf(4, quantile=0.5, cumulative=True)
            >>> bool(np.allclose(running[-1], median.sum(axis=0)))
            True
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))
            >>> roots = res.stability_profile()
            >>> roots.shape, bool(np.all(roots < 1.0)), bool(np.all(np.abs(roots - 0.5) < 0.2))
            ((3,), True, True)
            >>> QVAR(y, order=0).fit(quantiles=(0.5,)).stability_profile()
            array([0.])
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

        Example:
            The forecast is the level's coefficients applied to the last
            observation, and the forecasts order with the level:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((300, 2))
            >>> res = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))
            >>> point = res.forecast(0.5)
            >>> point.shape
            (2,)
            >>> by_hand = res.deterministic(0.5)[0] + res.coefficients(0.5)[0] @ y[-1]
            >>> bool(np.allclose(point, by_hand))
            True
            >>> bool(np.all(res.forecast(0.1) < point)), bool(np.all(point < res.forecast(0.9)))
            (True, True)
        """
        qi = self._index_of(quantile)
        n = self.endog.shape[0]
        det = {"n": [], "c": [1.0], "ct": [1.0, float(n + 1)]}[self.trend]
        out = np.asarray(det, dtype=np.float64) @ self.deterministics[qi]
        for lag in range(self.order):
            out = out + self.coefficient_stacks[qi, lag] @ self.endog[n - 1 - lag]
        return np.asarray(out, dtype=np.float64)

    def _specification(self) -> str:
        """Short specification label for display.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> QVAR(y, order=2).fit(quantiles=(0.5,))._specification()
            'QVAR(2)'
        """
        return f"QVAR({self.order})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with its own first-lag coefficient at the
            lowest and highest estimated levels and its ``R1`` at the
            lowest; the metadata lists the levels; the notes state the
            check-loss nature of the fit and its absent likelihood, the
            largest companion root across levels, the crossing share, the
            reduced-form reading of ``irf``, the one-step forecast rule,
            and the absence of standard errors.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> table = QVAR(y, order=1).fit(quantiles=(0.1, 0.5, 0.9))._summary_table()
            >>> table.title, len(table.rows), len(table.notes)
            ('QVAR(1) Results', 2, 6)
            >>> table.columns
            ('equation', 'own L1 (tau=0.1)', 'own L1 (tau=0.9)', 'R1 (tau=0.1)')
            >>> table.metadata[1]
            ('Levels', '0.1, 0.5, 0.9')
            >>> QVAR(y, order=0).fit(quantiles=(0.5,))._summary_table().rows[0][1:3]
            ('-', '-')
        """
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
    r"""Quantile vector autoregression, after Koenker and Bassett, equation by equation.

    Models the conditional :math:`\tau`-quantile of every variable as
    linear in the system's lags, with its own coefficient stack at every
    requested level,

    .. math::

       Q_\tau(y_{i,t} \mid y_{t-1}, \ldots, y_{t-p})
       = c_i(\tau)' d_t + \sum_{l=1}^{p} a_{i,l}(\tau)' y_{t-l},

    so the tails get their own dynamics rather than the mean's dynamics
    plus a symmetric error. Each equation at each level is one exact
    check-loss linear program, solved by HiGHS in primal form: the residual
    is split into its positive and negative parts, priced at :math:`\tau`
    and :math:`1 - \tau`, under :math:`X\beta + u - v = y`. Equations and
    levels separate because the check loss is additive across both, so a
    fit is :math:`k \times Q` independent programs over one design. No
    prior surface is offered: the base's shrinkage prior is a Gaussian
    belief about conditional-mean coefficients and has no meaning for a
    check loss.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _prior: The base slot, held at ``_NoPrior`` because no prior is offered.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order; zero fits the unconditional quantiles
            with deterministics only.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the panel, order, trend or names are
            malformed.
        DimensionError: If the sample cannot support the specification.

    See Also:
        * :class:`QVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the conditional-mean system on the same design.
        * :class:`~cultivars.univariate.conditional_variance.GARCH` -- a
          moving tail modelled through the variance instead.

    References:
        Koenker, R., & Bassett, G. (1978). Regression quantiles.
        *Econometrica*, 46(1), 33-50.

        White, H., Kim, T.-H., & Manganelli, S. (2015). VAR for VaR:
        Measuring tail dependence using multivariate regression quantiles.
        *Journal of Econometrics*, 187(1), 169-188.

        Adrian, T., Boyarchenko, N., & Giannone, D. (2019). Vulnerable
        growth. *American Economic Review*, 109(4), 1263-1289.

    Example:
        On a Gaussian VAR(1) the three levels share the same lag
        coefficients and differ in their intercepts, which order with the
        level:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> res = QVAR(y, order=1).fit(quantiles=(0.25, 0.5, 0.75))
        >>> res.coefficients(0.5).shape
        (1, 2, 2)
        >>> res.quantile_path("y1", 0.75).shape
        (199,)
        >>> spread = res.coefficient_stacks.max(axis=0) - res.coefficient_stacks.min(axis=0)
        >>> bool(spread.max() < 0.3)
        True
        >>> intercepts = res.deterministics[:, 0, 0]
        >>> bool(intercepts[0] < intercepts[1] < intercepts[2])
        True
        >>> QVAR(y, order=1, names=("a", "b")).fit(quantiles=(0.5,)).names
        ('a', 'b')
    """

    __slots__ = ()

    def fit(self, *, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> QVARResult:
        """Estimate every equation at every requested level.

        The levels are sorted ascending on the result whatever order they
        were given in; each is solved independently, so adding a level does
        not change the others.

        Args:
            quantiles: Distinct levels, each strictly inside ``(0, 1)``.
                The default brackets the distribution the way the
                growth-at-risk literature reads it: both tails and the
                median.

        Returns:
            The fitted :class:`QVARResult`.

        Raises:
            SpecificationError: If the levels are empty, repeated, or
                outside the open unit interval.
            NumericalError: If a program does not solve to optimality.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = QVAR(y, order=1)
            >>> model.fit().quantiles
            (0.1, 0.5, 0.9)
            >>> model.fit(quantiles=(0.9, 0.1)).quantiles
            (0.1, 0.9)
            >>> single = model.fit(quantiles=(0.5,))
            >>> bool(np.allclose(single.coefficients(0.5), model.fit().coefficients(0.5)))
            True
            >>> model.fit(quantiles=())
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: quantiles must name at least one level.
            >>> model.fit(quantiles=(0.5, 0.5))
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: quantiles must be distinct; got (0.5, 0.5).
            >>> model.fit(quantiles=(1.0,))
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: quantiles must lie in (0.0, 1.0); got 1.0.
        """
        return QVARResult._from_fit(self._fit_quantile(quantiles), self)
