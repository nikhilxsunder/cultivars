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
r"""Diebold-Yilmaz connectedness: who moves whom, in one table.

The spillover framework reads a fitted system's generalized forecast-error
variance decomposition as a weighted directed network. With
:math:`\Psi_h` the moving-average matrices of the fitted reduced form and
:math:`\Sigma_u` its innovation covariance, the share of variable
:math:`i`'s :math:`H`-horizon forecast-error variance associated with
shocks to :math:`j` is

.. math::

   \theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H}
   (e_i' \Psi_h \Sigma_u e_j)^2}
   {\sum_{h=0}^{H} e_i' \Psi_h \Sigma_u \Psi_h' e_i},
   \qquad
   \tilde\theta_{ij}(H) = 100\,\frac{\theta_{ij}(H)}{\sum_j \theta_{ij}(H)},

the generalized decomposition of Pesaran and Shin (1998), descended from
Koop, Pesaran and Potter (1996), which conditions on one shock at a time
without orthogonalizing and so assumes no variable ordering. The
row-normalized :math:`\tilde\theta` is the connectedness table: its
off-diagonal mass is connectedness, column sums are the directional
spillovers *to* others, row sums the spillovers *from* others, and their
difference ranks transmitters against receivers. One number -- the total
spillover index, the average off-diagonal mass -- summarizes how much of
the system's forecast uncertainty crosses variable boundaries at all, and
its movement through rolling windows is the framework's signature
picture: connectedness spiking into crises.

Two commitments shape the surface. First, this is a *view* of a fitted
reduced form, not an estimator, and it is built that way:
:class:`Spillover` constructs with any closed reduced-form result -- an
OLS VAR, a shrunk VAR, a sparse VAR estimated on a hundred series (the
Demirer-Diebold-Liu-Yilmaz pipeline), the regime view of a
Markov-switching VAR -- and composes rather than re-estimates, so the
network of a lasso VAR on wide panel is the same two lines as that of a
three-variable OLS VAR. Second, the one caveat is inherited from the
generalized decomposition itself and stated rather than hidden:
generalized shocks are correlated, so the raw shares do not sum to one,
the rows are normalized to 100, and the table is a relative attribution
rather than an additive decomposition. The summary says so, and says to
read the network as association, not causation.

Layout. :class:`Spillover` validates its source against the
``ClosedSystemResult`` protocol from ``_core`` -- ``names``, ``k_endog``,
``sigma_u`` and ``ma_representation`` are what it reads -- refuses a
horizon below one, and ``compute`` forms the decomposition directly from
the moving-average matrices and the innovation covariance, cumulating
:math:`\Psi_0, \ldots, \Psi_H` so that ``horizon=H`` matches
``generalized_fevd(H)[-1]`` on a VAR result. :class:`SpilloverResult`
carries the table and its directional readings and takes its summary from
``_SummaryMixin``. The estimators whose results feed it are
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`,
:mod:`~cultivars.multivariate.large_dim.bayesian`,
:mod:`~cultivars.multivariate.large_dim.sparse` and their neighbours; the
sparsity-based network reading of the same system is
:mod:`~cultivars.multivariate.large_dim.graphical`.

References:
    Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset return
    and volatility spillovers, with application to global equity markets.
    *Economic Journal*, 119(534), 158-171.

    Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive:
    Predictive directional measurement of volatility spillovers.
    *International Journal of Forecasting*, 28(1), 57-66.

    Demirer, M., Diebold, F. X., Liu, L., & Yilmaz, K. (2018). Estimating
    global bank network connectedness. *Journal of Applied Econometrics*,
    33(1), 1-15.

    Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response
    analysis in nonlinear multivariate models. *Journal of Econometrics*,
    74(1), 119-147.

    Pesaran, M. H., & Shin, Y. (1998). Generalized impulse response
    analysis in linear multivariate models. *Economics Letters*, 58(1),
    17-29.

Example:
    The same view on two estimators of one system: a four-variable VAR(1)
    in which ``y1`` drives everyone, read through OLS and through the
    lasso. Both name ``y1`` the transmitter, the lasso carries no more
    off-diagonal mass than OLS, and the identities hold on each:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.multivariate.large_dim.sparse import SparseVAR
    >>> rng = np.random.default_rng(0)
    >>> k, n = 4, 400
    >>> a = 0.4 * np.eye(k)
    >>> a[1:, 0] = 0.3
    >>> y = np.zeros((n, k))
    >>> for t in range(1, n):
    ...     y[t] = a @ y[t - 1] + rng.standard_normal(k)
    >>> ols = Spillover(VAR(y, order=1).fit(), horizon=10).compute()
    >>> lasso = Spillover(SparseVAR(y, order=1).fit(penalty="lasso"), horizon=10).compute()
    >>> ols.transmitters()[0], lasso.transmitters()[0]
    ('y1', 'y1')
    >>> bool(ols.net[0] > 20.0), bool(lasso.net[0] > 20.0)
    (True, True)
    >>> for spill in (ols, lasso):
    ...     print(bool(np.allclose(spill.table.sum(axis=1), 100.0)), bool(spill.net.sum() < 1e-9))
    True True
    True True
    >>> bool(lasso.total <= ols.total), "Diebold-Yilmaz Connectedness" in str(lasso.summary())
    (True, True)
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
    r"""A connectedness table and the directional readings it implies.

    All quantities are percentages of forecast-error variance from the
    row-normalized generalized decomposition,

    .. math::

       \theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H}
       (e_i' \Psi_h \Sigma_u e_j)^2}
       {\sum_{h=0}^{H} e_i' \Psi_h \Sigma_u \Psi_h' e_i},
       \qquad
       \tilde\theta_{ij}(H) = 100\,\frac{\theta_{ij}(H)}{\sum_j \theta_{ij}(H)},

    with :math:`\Psi_h` the moving-average matrices of the fitted system
    and :math:`\Sigma_u` its innovation covariance. The table is
    :math:`\tilde\theta`; its off-diagonal mass is connectedness, column
    sums are what each variable's shocks contribute to the others,
    row sums what each variable receives, and the difference ranks
    transmitters against receivers.

    Note:
        ``horizon`` is the largest moving-average lead in the sum, so the
        shares cumulate :math:`\Psi_0, \ldots, \Psi_H` -- :math:`H + 1`
        terms -- and equal ``100 * result.generalized_fevd(horizon)[-1]``
        on any result that offers that method. Rows of ``table`` sum to
        100 by construction; columns do not. ``net`` sums to zero across
        variables, and ``total`` equals the mean of ``from_others`` and
        of ``to_others``. Because generalized shocks are correlated the
        raw shares :math:`\theta_{ij}` do not sum to one, which is why the
        rows are normalized and why the table is a relative attribution
        rather than an additive decomposition.

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

    See Also:
        * :class:`Spillover` -- the view that produces this record.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- its ``generalized_fevd`` gives the same shares, unnormalized
          rows, at every horizon.

    References:
        Diebold, F. X., & Yilmaz, K. (2012). Better to give than to
        receive: Predictive directional measurement of volatility
        spillovers. *International Journal of Forecasting*, 28(1), 57-66.

        Pesaran, M. H., & Shin, Y. (1998). Generalized impulse response
        analysis in linear multivariate models. *Economics Letters*,
        58(1), 17-29.

    Example:
        A three-variable VAR(1) in which only ``y2`` drives ``y1``; the
        table names ``y2`` the transmitter and ``y1`` the receiver, and
        the accounting identities hold:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 3))
        >>> a = np.array([[0.5, 0.3, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        >>> for t in range(1, 300):
        ...     y[t] = a @ y[t - 1] + rng.standard_normal(3)
        >>> spill = Spillover(VAR(y, order=1).fit(), horizon=10).compute()
        >>> spill.names, spill.horizon, spill.k_endog, spill.table.shape
        (('y1', 'y2', 'y3'), 10, 3, (3, 3))
        >>> bool(np.allclose(spill.table.sum(axis=1), 100.0)), bool(abs(spill.net.sum()) < 1e-9)
        (True, True)
        >>> bool(spill.table[0, 1] > 5.0), bool(spill.table[1, 0] < 2.0)
        (True, True)
        >>> int(np.argmax(spill.net)), int(np.argmin(spill.net))
        (1, 0)
        >>> bool(abs(spill.total - spill.from_others.mean()) < 1e-9)
        True
    """

    names: tuple[str, ...]
    """Variable labels, indexing every axis of the arrays below."""
    horizon: int
    """The largest moving-average lead in the decomposition."""
    table: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` row-normalized shares in percent; rows sum to 100. Kept out of the repr."""
    total: float
    """Average off-diagonal mass of ``table``, in ``[0, 100]``."""
    to_others: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` column sums of the off-diagonal table. Kept out of the repr."""
    from_others: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` row sums of the off-diagonal table. Kept out of the repr."""
    net: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` ``to_others - from_others``; sums to zero. Kept out of the repr."""
    net_pairwise: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` antisymmetric ``table.T - table``. Kept out of the repr."""

    @property
    def k_endog(self) -> int:
        """Number of variables in the network.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((150, 4))
            >>> Spillover(VAR(y, order=1).fit()).compute().k_endog
            4
        """
        return len(self.names)

    def transmitters(self) -> tuple[str, ...]:
        """Variables that transmit more variance than they receive, ranked.

        Returns:
            The names with positive ``net``, largest first; empty when no
            variable is a net transmitter.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 3))
            >>> a = np.array([[0.5, 0.3, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
            >>> for t in range(1, 300):
            ...     y[t] = a @ y[t - 1] + rng.standard_normal(3)
            >>> spill = Spillover(VAR(y, order=1).fit(), horizon=10).compute()
            >>> spill.transmitters()[0]
            'y2'
            >>> "y1" in spill.transmitters()
            False
        """
        order = np.argsort(self.net)[::-1]
        return tuple(self.names[i] for i in order if self.net[i] > 0.0)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per variable with its ``to``, ``from`` and signed
            ``net`` spillovers; the metadata carries the dimension, the
            horizon and the total index; the notes state the total in
            words, the decomposition's provenance and its relative-share
            caveat, and the association-not-causation reading.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((150, 3))
            >>> spill = Spillover(VAR(y, order=1).fit(), horizon=5).compute()
            >>> table = spill._summary_table()
            >>> table.title, len(table.rows), len(table.notes)
            ('Diebold-Yilmaz Connectedness', 3, 3)
            >>> table.columns
            ('variable', 'to others', 'from others', 'net')
            >>> table.metadata[1]
            ('Horizon', '5')
        """
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
    generalized variance decomposition as a directed network. Nothing is
    re-estimated; the view composes with whatever estimator produced the
    result, so the connectedness of a lasso VAR on a hundred series is the
    same two lines as that of a three-variable OLS VAR.

    Attributes:
        _result: The fitted closed reduced-form result read from.
        _horizon: The largest moving-average lead in the decomposition.

    Args:
        result: A fitted closed reduced-form result -- one exposing
            ``names``, ``k_endog``, ``sigma_u`` and ``ma_representation``.
        horizon: Forecast horizon the decomposition is read at, at least
            1; ten to twelve steps is the literature's habit.

    Raises:
        SpecificationError: If the result is not a closed system or the
            horizon is not positive.

    See Also:
        * :class:`SpilloverResult` -- the record ``compute`` returns.
        * :class:`~cultivars.multivariate.large_dim.sparse.SparseVAR` -- the
          estimator for the wide panels the framework was built for.
        * :class:`~cultivars.multivariate.large_dim.graphical.GraphicalVAR`
          -- the sparsity-based network readings of the same system.

    References:
        Diebold, F. X., & Yilmaz, K. (2009). Measuring financial asset
        return and volatility spillovers, with application to global
        equity markets. *Economic Journal*, 119(534), 158-171.

        Demirer, M., Diebold, F. X., Liu, L., & Yilmaz, K. (2018).
        Estimating global bank network connectedness. *Journal of
        Applied Econometrics*, 33(1), 1-15.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
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
        >>> Spillover(y, horizon=10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: Spillover reads a fitted closed reduced-form ...
    """

    __slots__ = ("_horizon", "_result")

    def __init__(self, result: ClosedSystemResult, *, horizon: int = 10) -> None:
        """Validate the source surface and the horizon.

        Args:
            result: A fitted closed reduced-form result.
            horizon: Forecast horizon the decomposition is read at.

        Raises:
            SpecificationError: If the result does not satisfy the
                closed-system protocol or ``horizon`` is below 1.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> fit = VAR(np.random.default_rng(0).standard_normal((150, 2)), order=1).fit()
            >>> Spillover(fit)._horizon
            10
            >>> Spillover(fit, horizon=0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be at least 1; got 0.
        """
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
        r"""Read the connectedness table off the generalized decomposition.

        Forms :math:`\Psi_h \Sigma_u e_j / \sqrt{\sigma_{jj}}` for every
        lead and every shock, cumulates the squares over
        :math:`h = 0, \ldots, H`, normalizes each row to 100 and reads the
        directional sums off the off-diagonal.

        Returns:
            The :class:`SpilloverResult`, all entries in percent.

        Example:
            The table agrees with the VAR result's own generalized
            decomposition at the same horizon, and connectedness grows
            with the horizon as shocks propagate:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 3))
            >>> a = np.array([[0.5, 0.3, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
            >>> for t in range(1, 300):
            ...     y[t] = a @ y[t - 1] + rng.standard_normal(3)
            >>> fit = VAR(y, order=1).fit()
            >>> spill = Spillover(fit, horizon=10).compute()
            >>> bool(np.allclose(spill.table, 100.0 * fit.generalized_fevd(10)[-1]))
            True
            >>> short = Spillover(fit, horizon=1).compute()
            >>> bool(short.total < spill.total)
            True
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
