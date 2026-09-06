# filepath: /src/cultivars/multivariate/large_dim/spillover.py
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

"""Diebold-Yilmaz connectedness: who moves whom, in one table.

The spillover framework reads a fitted system's generalized forecast-error
variance decomposition (Koop-Pesaran-Shin, so no variable ordering is
assumed) as a weighted directed network: the ``(i, j)`` cell is the share of
variable ``i``'s forecast-error variance associated with shocks to ``j``,
off-diagonal mass is connectedness, and row and column sums become the
"from" and "to" directional spillovers whose difference ranks transmitters
against receivers. One number -- the total spillover index -- summarizes how
much of the system's forecast uncertainty crosses variable boundaries at
all, and its movement through rolling windows is the framework's signature
picture: connectedness spiking into crises.

This is a *view* of a fitted reduced form, not an estimator, and it is built
that way: :class:`Spillover` constructs with any closed reduced-form result
-- an OLS VAR, a shrunk VAR, a sparse VAR estimated on a hundred series
(Demirer-Diebold-Liu-Yilmaz), the regime view of a Markov-switching VAR --
and composes rather than re-estimates. The one caveat is inherited from the
generalized decomposition itself: generalized shocks are correlated, so rows
are normalized to sum to one, and the shares are a relative attribution
rather than an additive decomposition. Read the network as association, not
causation.

References:
    Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset return
        and volatility spillovers, with application to global equity
        markets. *Economic Journal*, 119(534), 158-171.
    Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive:
        Predictive directional measurement of volatility spillovers.
        *International Journal of Forecasting*, 28(1), 57-66.
    Demirer, M., Diebold, F. X., Liu, L., & Yilmaz, K. (2018). Estimating
        global bank network connectedness. *Journal of Applied
        Econometrics*, 33(1), 1-15.
    Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response
        analysis in nonlinear multivariate models. *Journal of
        Econometrics*, 74(1), 119-147.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import ClosedSystemResult, SummaryTable
from ..._internals import _SummaryMixin
from ...exceptions import SpecificationError

__all__ = ["Spillover", "SpilloverResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpilloverResult(_SummaryMixin):
    """A connectedness table and the directional readings it implies.

    All quantities are percentages of ``h``-step forecast-error variance,
    from the row-normalized generalized decomposition.

    Attributes:
        names: Variable labels, indexing every axis below.
        horizon: The forecast horizon the decomposition was read at.
        table: ``(k, k)`` connectedness table; entry ``[i, j]`` is the share
            of ``i``'s forecast-error variance associated with shocks to
            ``j``. Rows sum to 100.
        total: The total spillover index -- average off-diagonal mass, in
            ``[0, 100]``.
        to_others: ``(k,)`` directional spillover *to* others: what shocks
            to each variable contribute to everyone else's variance.
        from_others: ``(k,)`` directional spillover *from* others: how much
            of each variable's variance arrives from elsewhere.
        net: ``to_others - from_others``; positive ranks a variable as a
            net transmitter.
        net_pairwise: ``(k, k)`` antisymmetric net pairwise spillovers,
            ``table[j, i] - table[i, j]`` read as ``i``'s net transmission
            to ``j``.
    """

    names: tuple[str, ...]
    horizon: int
    table: npt.NDArray[np.float64] = field(repr=False)
    total: float
    to_others: npt.NDArray[np.float64] = field(repr=False)
    from_others: npt.NDArray[np.float64] = field(repr=False)
    net: npt.NDArray[np.float64] = field(repr=False)
    net_pairwise: npt.NDArray[np.float64] = field(repr=False)

    @property
    def k_endog(self) -> int:
        """Number of variables in the network."""
        return len(self.names)

    def transmitters(self) -> tuple[str, ...]:
        """Variables that transmit more variance than they receive, ranked."""
        order = np.argsort(self.net)[::-1]
        return tuple(self.names[i] for i in order if self.net[i] > 0.0)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = tuple(
            (
                name,
                f"{self.to_others[i]:.1f}",
                f"{self.from_others[i]:.1f}",
                f"{self.net[i]:+.1f}",
            )
            for i, name in enumerate(self.names)
        )
        notes = [
            f"Total spillover index: {self.total:.1f}% of {self.horizon}-step "
            "forecast-error variance crosses variable boundaries.",
            "Shares come from the row-normalized generalized decomposition "
            "(Koop-Pesaran-Shin): no variable ordering is assumed, and "
            "because generalized shocks are correlated the shares are a "
            "relative attribution, not an additive decomposition.",
            "Read the network as association, not causation; net > 0 marks "
            "a net transmitter at this horizon.",
        ]
        return SummaryTable(
            title="Diebold-Yilmaz Connectedness",
            metadata=(
                ("Variables", f"{self.k_endog}"),
                ("Horizon", f"{self.horizon}"),
                ("Total spillover", f"{self.total:.1f}%"),
            ),
            columns=("variable", "to others", "from others", "net"),
            rows=rows,
            notes=tuple(notes),
        )


class Spillover:
    """Diebold-Yilmaz connectedness of any fitted closed reduced form.

    Not an estimator: constructs with a fitted result -- anything exposing
    the closed-system surface, from an OLS VAR to a sparse hundred-variable
    system to one regime of a Markov-switching VAR -- and reads its
    generalized variance decomposition as a directed network.

    Args:
        result: A fitted closed reduced-form result.
        horizon: Forecast horizon the decomposition is read at; ten to
            twelve steps is the literature's habit.

    Raises:
        SpecificationError: If the result is not a closed system or the
            horizon is not positive.

    Example:
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 3))
        >>> a = np.array([[0.5, 0.3, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        >>> for t in range(1, 300):
        ...     y[t] = a @ y[t - 1] + rng.standard_normal(3)
        >>> spill = Spillover(VAR(y, order=1).fit(), horizon=10).compute()
        >>> spill.table.shape
        (3, 3)
        >>> bool(np.all(np.abs(spill.table.sum(axis=1) - 100.0) < 1e-8))
        True
    """

    __slots__ = ("_horizon", "_result")

    def __init__(self, result: ClosedSystemResult, *, horizon: int = 10) -> None:
        """Validate the source surface and the horizon."""
        if not isinstance(result, ClosedSystemResult):
            raise SpecificationError(
                "Spillover reads a fitted closed reduced-form result exposing "
                "names, sigma_u, and ma_representation; got "
                f"{type(result).__name__}."
            )
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        self._result = result
        self._horizon = int(horizon)

    def compute(self) -> SpilloverResult:
        """Read the connectedness table off the generalized decomposition.

        Returns:
            The :class:`SpilloverResult`, all entries in percent.
        """
        result = self._result
        k = result.k_endog
        psi = result.ma_representation(self._horizon)
        sigma = result.sigma_u
        scale = np.sqrt(np.diag(sigma))
        theta = np.stack([psi @ sigma[:, j] / scale[j] for j in range(k)], axis=-1)
        contribution = np.cumsum(theta**2, axis=0)[-1]
        shares = 100.0 * contribution / contribution.sum(axis=1, keepdims=True)
        off_diagonal = shares - np.diag(np.diag(shares))
        to_others = off_diagonal.sum(axis=0)
        from_others = off_diagonal.sum(axis=1)
        return SpilloverResult(
            names=tuple(result.names),
            horizon=self._horizon,
            table=shares,
            total=float(off_diagonal.sum() / k),
            to_others=to_others,
            from_others=from_others,
            net=to_others - from_others,
            net_pairwise=shares.T - shares,
        )
