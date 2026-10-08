# filepath: /src/cultivars/_internals/_means.py
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

"""Mean layers: how a conditional-variance model turns a draw into residuals.

A conditional-variance likelihood needs exactly three things from the mean
specification -- how many parameters it consumes, what residuals a given draw
implies, and where to start the search. Everything else about the variance
families is indifferent to whether that mean is a regression on lags or a
recursion with a moving-average block.

Factoring those three out is what lets one variance objective serve both. Before
this split, the objective held a design matrix and formed residuals as
``target - design @ mean``, which silently assumed the mean was linear in
observables -- true for a constant and for pure autoregressive lags, false the
moment a moving-average term appears, because a lagged *residual* is not a
column you can build in advance. Adding an MA block by widening the design was
never an option; it needed a different residual map, which is precisely what a
layer is.

Two layers cover the group:

1. :class:`_LinearMean` is the regression case. Residuals come from one matrix
   product and the coefficients are unconstrained, since a mean that is
   estimated jointly with a variance does not need its autoregressive roots
   pinned inside the unit circle for the recursion to be evaluable.
2. :class:`_ArmaMean` is the recursion case. Residuals are produced one period
   at a time, so an explosive draw would feed itself and overflow -- which is
   why this layer reparameterizes both blocks through the partial
   autocorrelations. Stationarity and invertibility then hold by construction
   and the optimizer searches an unconstrained space.

The layer owns its target rather than receiving it. An autoregressive-moving-
average residual recursion reads levels from *before* the first modelled
observation, so it needs the untrimmed series; handing it only the trimmed
target and asking for residuals would either lose those lags or require the
caller to know the trim, and the caller is the one place that should not have
to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import ols, pack_stationary, unpack_stationary
from ._coefficients import _MeanCoefficients


class _MeanLayer(ABC):
    """The conditional-mean half of a conditional-variance likelihood.

    Concrete layers are frozen and hold whatever data their residual map needs,
    so an objective can ask for residuals with nothing but a parameter block.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def target(self) -> npt.NDArray[np.float64]:
        """The observations the variance recursion is evaluated over."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Width of this layer's slice of the parameter vector."""

    @abstractmethod
    def residuals(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Residuals implied by a draw, aligned with :attr:`target`.

        Args:
            params: This layer's slice of the parameter vector, in whatever
                coordinates the layer searches.

        Returns:
            An array the same length as :attr:`target`.
        """

    @abstractmethod
    def start(self) -> npt.NDArray[np.float64]:
        """A warm start for this layer's block, in search coordinates."""

    @abstractmethod
    def unpack(self, params: npt.NDArray[np.float64]) -> _MeanCoefficients:
        """Interpret a draw as named coefficients for reporting.

        Args:
            params: This layer's slice of the parameter vector.

        Returns:
            The intercept, autoregressive block, and moving-average block in
            their natural coordinates rather than the searched ones.
        """

    def fitted(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means implied by a draw, aligned with :attr:`target`.

        Defaults to the difference against the residuals, which is the only
        thing a recursion can offer. A layer that computes the mean directly
        should override this rather than let the difference round-trip: for a
        matrix product ``t - (t - x)`` is not bit-identical to ``x``, and a
        reported fitted value that disagrees with its own definition in the
        last unit in the last place is the kind of drift that makes a
        regression test fail for no reason anyone can find.

        Args:
            params: This layer's slice of the parameter vector.

        Returns:
            An array the same length as :attr:`target`.
        """
        return self.target - self.residuals(params)

    def initial_residuals(self) -> npt.NDArray[np.float64]:
        """Residuals at the warm start, used to seed the variance backcast."""
        return self.residuals(self.start())


@dataclass(frozen=True, slots=True)
class _LinearMean(_MeanLayer):
    """A mean that is linear in a fixed design: a constant and lagged levels.

    Attributes:
        endog_target: The trimmed series the design explains.
        design: Columns of the mean regression, intercept first when present.
            May have zero columns, which is the zero-mean specification.
        include_const: Whether the first design column is the intercept.
    """

    endog_target: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    include_const: bool

    @property
    def target(self) -> npt.NDArray[np.float64]:
        """The observations the variance recursion is evaluated over."""
        return self.endog_target

    @property
    def n_parameters(self) -> int:
        """Number of design columns."""
        return int(self.design.shape[1])

    def residuals(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Residuals from one matrix product; the zero-column case is the series."""
        if not self.n_parameters:
            return self.endog_target
        return self.endog_target - self.design @ params

    def fitted(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The design times the coefficients, computed rather than differenced."""
        if not self.n_parameters:
            return np.zeros(self.endog_target.shape[0], dtype=np.float64)
        return self.design @ params

    def start(self) -> npt.NDArray[np.float64]:
        """Ordinary least squares on the design, or an empty block."""
        if not self.n_parameters:
            return np.zeros(0, dtype=np.float64)
        return ols(self.design, self.endog_target)[0]

    def unpack(self, params: npt.NDArray[np.float64]) -> _MeanCoefficients:
        """Split the coefficient vector into an intercept and lag weights."""
        if self.include_const:
            return _MeanCoefficients(
                const=float(params[0]),
                ar=np.asarray(params[1:], dtype=np.float64),
                ma=np.zeros(0, dtype=np.float64),
            )
        return _MeanCoefficients(
            const=None,
            ar=np.asarray(params, dtype=np.float64),
            ma=np.zeros(0, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class _ARMAMean(_MeanLayer):
    """A mean produced by a conditional-sum-of-squares ARMA recursion.

    The recursion conditions on the first ``max(p, q)`` observations: lagged
    levels before that point are taken from the untrimmed series, and lagged
    residuals before it are set to zero, which is the standard conditional
    convention. Exact maximum likelihood through the state-space form is not
    available here -- it assumes a constant innovation variance, and the whole
    point of the surrounding model is that the variance is not constant.

    Both blocks are searched through the partial autocorrelations, so the
    autoregressive roots stay inside the unit circle and the moving-average
    roots stay outside it for every draw the optimizer can propose. That is not
    cosmetic: the residual recursion is fed by its own past output, so an
    explosive draw does not merely score badly, it overflows.

    Attributes:
        endog: The untrimmed series.
        p: Autoregressive order.
        q: Moving-average order.
        include_const: Whether an intercept is estimated.
    """

    endog: npt.NDArray[np.float64]
    p: int
    q: int
    include_const: bool

    @property
    def burn(self) -> int:
        """Leading observations the recursion conditions on."""
        return max(self.p, self.q)

    @property
    def target(self) -> npt.NDArray[np.float64]:
        """The series past the conditioning block."""
        return self.endog[self.burn :]

    @property
    def n_parameters(self) -> int:
        """Intercept plus both blocks."""
        return int(self.include_const) + self.p + self.q

    def _split(
        self, params: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map search coordinates to an intercept and two natural-scale blocks."""
        offset = int(self.include_const)
        const = float(params[0]) if self.include_const else 0.0
        ar = unpack_stationary(np.asarray(params[offset : offset + self.p], dtype=np.float64))
        ma = unpack_stationary(np.asarray(params[offset + self.p :], dtype=np.float64))
        return const, ar, ma

    def residuals(self, params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Run the conditional recursion over the untrimmed series.

        Args:
            params: ``[const?, psi_ar, psi_ma]`` in unconstrained coordinates.

        Returns:
            Residuals for the trimmed sample, length ``len(endog) - burn``.
        """
        const, ar, ma = self._split(params)
        y = self.endog
        n, burn = y.shape[0], self.burn
        eps = np.zeros(n, dtype=np.float64)
        for t in range(burn, n):
            mu = const
            for i in range(self.p):
                mu += ar[i] * y[t - i - 1]
            for j in range(self.q):
                mu += ma[j] * eps[t - j - 1]
            eps[t] = y[t] - mu
        return eps[burn:]

    def start(self) -> npt.NDArray[np.float64]:
        """Least-squares autoregression for the level blocks, zero for the MA.

        A zero moving-average start is deliberate rather than lazy: it places
        the search at an ARMA whose MA polynomial is the identity, which is the
        nested autoregression, so the first optimizer steps improve on a model
        that is already sensible instead of on a random one.
        """
        blocks: list[npt.NDArray[np.float64]] = []
        y, burn = self.endog, self.burn
        if self.p:
            design = np.column_stack(
                ([np.ones(y.shape[0] - burn)] if self.include_const else [])
                + [y[burn - i - 1 : y.shape[0] - i - 1] for i in range(self.p)]
            )
            coeffs = ols(design, y[burn:])[0]
            if self.include_const:
                blocks.append(coeffs[:1])
            blocks.append(pack_stationary(np.asarray(coeffs[int(self.include_const) :])))
        elif self.include_const:
            blocks.append(np.array([float(np.mean(y[burn:]))]))
        blocks.append(np.zeros(self.q, dtype=np.float64))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def unpack(self, params: npt.NDArray[np.float64]) -> _MeanCoefficients:
        """Report the intercept and both blocks on their natural scale."""
        const, ar, ma = self._split(params)
        return _MeanCoefficients(
            const=const if self.include_const else None,
            ar=ar,
            ma=ma,
        )
