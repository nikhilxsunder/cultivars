# filepath: /src/cultivars/multivariate/nonlinear/time_varying.py
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
r"""Time-varying-parameter VARs: drift instead of switching, posteriors instead of dates.

Where the regime families let parameters jump between a few states, here
they drift: the stacked coefficient vector follows a random walk, and
under stochastic volatility so do the log variances,

.. math::

   y_t = X_t' \beta_t + u_t, \qquad
   \beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q),
   \qquad
   u_t \sim N(0, \Sigma_t), \quad
   \Sigma_t = A^{-1} H_t A^{-\top}, \quad
   h_t = h_{t-1} + \zeta_t,

so every date has its own VAR and its own covariance (the homoskedastic
model holds :math:`\Sigma_t = \Sigma`). This is the Cogley-Sargent /
Primiceri (2005) workhorse for questions like whether monetary policy
transmission itself has changed -- questions a constant-parameter model
cannot even pose.

Two commitments shape the surface. First, estimation is Bayesian by
necessity, not preference: an unrestricted coefficient path has more
parameters than observations, and what disciplines it is the prior on how
fast parameters may drift. Following Primiceri, a training sample is
split off the front, its OLS estimates calibrate the coefficient prior and
the drift scale ``k_drift`` (his :math:`k_Q`) and the volatility scale
``k_vol`` (his :math:`k_W`), and it is then discarded rather than used
twice. Those two scales are the model's findings in disguise: at the
defaults of ``0.01`` a coefficient that drifts by several tenths, or a
volatility that triples, over a few hundred observations is reported as
nearly flat, so the results compare each path's movement with its own
band and the user is expected to vary the scales. The sampler is Gibbs on
the package's own state-space substrate: every coefficient-path draw is
one Durbin-Koopman simulation-smoother call, the covariance blocks are
conjugate inverse-Wisharts, and the stochastic-volatility block is
Kim-Shephard-Chib, with the log-volatility innovation covariance
:math:`W` diagonal as in Cogley and Sargent rather than Primiceri's full
matrix. Second, under SV the innovation covariance is parameterized
through a constant unit-lower-triangular :math:`A` in the order of
``names``, which is why the SV model's structural interpretation is
settled at estimation time (see :class:`TVPSVAR`) and why its
``sigma_u`` is the time-averaged covariance rather than any one date's.

Being posteriors, the results report uncertainty rather than test
statistics: coefficient and volatility paths come with posterior bands,
and there is deliberately no likelihood, information criterion, or
parameter count on these results -- a Gibbs posterior has none to report,
and numbers shaped like them would be wrong. Structural responses at a
date freeze that date's posterior means and are read as the transmission
prevailing there.

Layout. :class:`TVPVAR` and :class:`TVPVARSV` validate on
``_TimeVaryingVectorAutoRegressionModel`` in ``_internals``, which
resolves the training split, builds the Kronecker-stacked observation
matrices, calibrates the priors by OLS on the training rows, and in
``_fit_tvp`` runs the sampler: a ``_LinearGaussianStateSpace`` from the
state-space layer supplies the simulation smoother, ``_draw_inverse_wishart``
and ``_draw_triangular_volatility_block`` from ``_core._samplers`` the
covariance and volatility blocks, and a ``_TimeVaryingFit`` is packed.
:class:`TVPVARResult` and :class:`TVPVARSVResult` take their summary from
``_SummaryMixin`` and their chain diagnostics from ``_ConvergenceMixin``;
:class:`TVPSVAR` reads either into a :class:`TVPSVARResult` through
:func:`~cultivars._core.companion_matrix`. The constant-coefficient
posteriors these paths relax are in
:mod:`~cultivars.multivariate.large_dim.bayesian` and
:mod:`~cultivars.multivariate.large_dim.volatility`; the models whose
parameters switch rather than drift are the rest of this package and
:mod:`~cultivars.multivariate.regime_switching.markov_switching`.

References:
    Primiceri, G. E. (2005). Time varying structural vector autoregressions
    and monetary policy. *Review of Economic Studies*, 72(3), 821-852.

    Del Negro, M., & Primiceri, G. E. (2015). Time varying structural
    vector autoregressions and monetary policy: A corrigendum. *Review of
    Economic Studies*, 82(4), 1342-1345.

    Cogley, T., & Sargent, T. J. (2005). Drifts and volatilities: Monetary
    policies and outcomes in the post WWII US. *Review of Economic
    Dynamics*, 8(2), 262-302.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *Review of
    Economic Studies*, 65(3), 361-393.

    Durbin, J., & Koopman, S. J. (2002). A simple and efficient simulation
    smoother for state space time series analysis. *Biometrika*, 89(3),
    603-616.

Example:
    Drift and volatility change in one system: the own-lag coefficient of
    ``y1`` climbs over the sample while the shocks grow louder. With the
    scales loosened the SV fit sees both, and the structural impact of
    the first shock at the end of the sample exceeds its impact at the
    start:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> y = np.zeros((n, 2))
    >>> for t in range(1, n):
    ...     a = 0.2 + 0.6 * t / n
    ...     scale = 0.5 + 1.0 * t / n
    ...     y[t, 0] = a * y[t - 1, 0] + scale * rng.standard_normal()
    ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.3 * y[t - 1, 0] + rng.standard_normal()
    >>> res = TVPVARSV(y, order=1, training=40).fit(
    ...     n_draws=60, n_burn=20, k_drift=0.2, k_vol=0.1, seed=0
    ... )
    >>> path = res.coefficient_path("y1", "y1.L1")[:, 1]
    >>> vol = res.volatility_path("y1")[:, 1]
    >>> bool(path[-1] - path[0] > 0.2), bool(vol[-1] / vol[0] > 1.5)
    (True, True)
    >>> svar = TVPSVAR(res).identify()
    >>> bool(svar.impact(res.nobs - 1)[0, 0] > svar.impact(0)[0, 0])
    True
    >>> svar.irf(4, t=res.nobs - 1).shape
    (5, 2, 2)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, companion_matrix
from ..._internals import (
    _ConvergenceMixin,
    _SummaryMixin,
    _TimeVaryingVectorAutoRegressionModel,
)
from ...exceptions import SpecificationError

__all__ = [
    "TVPSVAR",
    "TVPVAR",
    "TVPVARSV",
    "TVPSVARResult",
    "TVPVARResult",
    "TVPVARSVResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPVARResult(_SummaryMixin, _ConvergenceMixin):
    r"""A fitted time-varying-parameter VAR, homoskedastic innovations.

    The posterior of

    .. math::

       y_t = X_t' \beta_t + u_t, \qquad u_t \sim N(0, \Sigma),
       \qquad
       \beta_t = \beta_{t-1} + \eta_t, \qquad \eta_t \sim N(0, Q),

    with :math:`X_t` the Kronecker-stacked design and :math:`\beta_t` the
    :math:`D = k\,w` coefficients of all equations stacked equation-major,
    reached by Gibbs sampling: each coefficient path is one Durbin-Koopman
    simulation-smoother draw given :math:`(Q, \Sigma)`, and the two
    covariances are conjugate inverse-Wishart draws given the path. Every
    coefficient is a path: the posterior over :math:`\beta_t` is summarized
    by its mean and a 68% band, with the kept draws retained so downstream
    consumers -- the structural wrapper, custom summaries -- can propagate
    posterior uncertainty rather than plugging in a point.

    Deliberately absent: ``llf``, ``n_params``, information criteria. A
    Gibbs posterior has none of them, and this result will not counterfeit
    the comparison surface of a likelihood fit.

    Note:
        The first ``training`` rows calibrated the priors -- the initial
        state from their OLS estimate, the drift covariance :math:`Q` from
        ``k_drift`` times that estimate's covariance -- and were then
        discarded, so every path here is indexed on the ``nobs`` rows that
        follow. How much the coefficients are allowed to move is a prior
        choice, not a finding: under Primiceri's default ``k_drift = 0.01``
        a slow deterministic drift of several tenths over two hundred
        observations is shrunk to a few hundredths, and the drift note in
        the summary compares the posterior mean's movement with its band
        for that reason. The point summaries (``sigma_u``, ``state_cov``,
        ``resid``, ``fittedvalues``) are at posterior means. The draws are
        a Markov chain; :meth:`convergence` reads them.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        training: Rows consumed by the training prior and then discarded.
        nobs: Estimation rows after the training split.
        beta_mean: ``(nobs, D)`` posterior mean coefficient path, stacked
            equation-major.
        beta_low: ``(nobs, D)`` posterior 16th percentile path.
        beta_high: ``(nobs, D)`` posterior 84th percentile path.
        beta_draws: ``(S, nobs, D)`` kept coefficient-path draws.
        state_cov: ``(D, D)`` posterior mean drift covariance ``Q``.
        sigma_u: ``(k, k)`` posterior mean innovation covariance.
        resid: Residuals at the posterior mean path.
        fittedvalues: One-step means at the posterior mean path.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    See Also:
        * :class:`TVPVAR` -- the model that produces this record.
        * :class:`TVPVARSVResult` -- the same paths with drifting
          volatility.
        * :class:`TimeVaryingSVAR` -- the recursive structural reading of
          either.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVARResult` --
          the constant-coefficient posterior these paths relax.

    References:
        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

        Cogley, T., & Sargent, T. J. (2005). Drifts and volatilities:
        Monetary policies and outcomes in the post WWII US. *Review of
        Economic Dynamics*, 8(2), 262-302.

    Example:
        A bivariate system whose first own-lag coefficient drifts linearly
        from 0.2 to 0.8; with a loose drift prior the posterior mean path
        climbs by several tenths, its band brackets the path throughout,
        and the local companion root rises with it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((240, 2))
        >>> phi = np.linspace(0.2, 0.8, 240)
        >>> for t in range(1, 240):
        ...     y[t, 0] = phi[t] * y[t - 1, 0] + rng.standard_normal()
        ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.3 * y[t - 1, 0] + rng.standard_normal()
        >>> res = TVPVAR(y, order=1, training=40).fit(
        ...     n_draws=60, n_burn=20, k_drift=0.3, seed=0
        ... )
        >>> res.training, res.nobs, res.n_kept, res.beta_mean.shape, res.beta_draws.shape
        (40, 199, 20, (199, 6), (20, 199, 6))
        >>> path = res.coefficient_path("y1", "y1.L1")
        >>> bool(path[-1, 1] - path[0, 1] > 0.25), bool(np.all(path[:, 0] < path[:, 2]))
        (True, True)
        >>> roots = res.stability_path()
        >>> bool(roots[-1] > roots[0]), bool(roots.max() < 1.0)
        (True, True)
        >>> res.sigma_u.shape, res.state_cov.shape
        ((2, 2), (6, 6))
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed panel, ``(nobs_total, k)``, training rows included. Kept out of the repr."""
    names: tuple[str, ...]
    """Variable labels, in column order."""
    order: int
    """The autoregressive order ``p``."""
    trend: str
    """The deterministic specification: ``"n"``, ``"c"`` or ``"ct"``."""
    training: int
    """Rows split off the front to calibrate the priors, then discarded."""
    nobs: int
    """Estimation rows after the training split and the ``p`` presample rows."""
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, D)`` posterior mean coefficient path, equation-major. Kept out of the repr."""
    beta_low: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, D)`` posterior 16th-percentile path. Kept out of the repr."""
    beta_high: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, D)`` posterior 84th-percentile path. Kept out of the repr."""
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, nobs, D)`` kept coefficient-path draws. Kept out of the repr."""
    state_cov: npt.NDArray[np.float64] = field(repr=False)
    """``(D, D)`` posterior mean drift covariance ``Q``. Kept out of the repr."""
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` posterior mean innovation covariance. Kept out of the repr."""
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` residuals at the posterior mean path. Kept out of the repr."""
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` one-step means at the posterior mean path. Kept out of the repr."""
    n_draws: int
    """Total sampler iterations, burn-in included."""
    n_burn: int
    """Iterations discarded from the start."""
    thin: int
    """Every ``thin``-th post-burn iteration is kept."""

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> TVPVAR(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0).k_endog
            3
        """
        return len(self.names)

    @property
    def n_kept(self) -> int:
        """Posterior draws retained after burn-in and thinning.

        ``ceil((n_draws - n_burn) / thin)``, the leading axis of every
        ``*_draws`` array.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=1, training=30).fit(n_draws=30, n_burn=10, thin=3, seed=0)
            >>> res.n_kept, res.beta_draws.shape[0]
            (7, 7)
        """
        return int(self.beta_draws.shape[0])

    @property
    def _n_deterministic(self) -> int:
        """Deterministic columns per equation.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = TVPVAR(y, order=1, training=30, trend="ct")
            >>> model.fit(n_draws=12, n_burn=4, seed=0)._n_deterministic
            2
        """
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    @property
    def _width(self) -> int:
        """Regressors per equation, ``w = n_det + k p``; the state has ``k w`` entries.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=2, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> res._width, res.beta_mean.shape[1] == res.k_endog * res._width
            (5, True)
        """
        return self._n_deterministic + self.k_endog * self.order

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design order.

        Returns:
            Deterministic labels first, then ``"{name}.L{lag}"`` lag by lag;
            equation ``i`` occupies state entries ``i * w .. (i + 1) * w - 1``
            in this order.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=2, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> res._regressor_labels()
            ('const', 'y1.L1', 'y2.L1', 'y1.L2', 'y2.L2')
        """
        det = ("const", "trend")[: self._n_deterministic]
        lags = tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)
        return (*det, *lags)

    def coefficient_path(self, equation: str, regressor: str) -> npt.NDArray[np.float64]:
        """One coefficient's posterior path: 16th percentile, mean, 84th.

        Args:
            equation: An endogenous variable.
            regressor: A per-equation regressor label -- ``"const"``,
                ``"trend"``, or ``"{name}.L{lag}"``.

        Returns:
            An ``(nobs, 3)`` array with columns ``(low, mean, high)``.

        Raises:
            SpecificationError: If the equation or regressor is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> path = res.coefficient_path("y2", "y1.L1")
            >>> path.shape, bool(np.all(path[:, 0] <= path[:, 1]))
            ((99, 3), True)
            >>> bool(np.all(path[:, 1] <= path[:, 2]))
            True
            >>> bool(np.allclose(path[:, 1], res.beta_mean[:, res._width + 1]))
            True
            >>> res.coefficient_path("y1", "y1.L2")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown regressor 'y1.L2'; expected one of ...
        """
        if equation not in self.names:
            raise SpecificationError(
                f"unknown variable {equation!r}; expected one of {self.names}."
            )
        labels = self._regressor_labels()
        if regressor not in labels:
            raise SpecificationError(f"unknown regressor {regressor!r}; expected one of {labels}.")
        column = self.names.index(equation) * self._width + labels.index(regressor)
        return np.column_stack(
            [self.beta_low[:, column], self.beta_mean[:, column], self.beta_high[:, column]]
        )

    def coefficients_at(self, t: int) -> npt.NDArray[np.float64]:
        """The posterior mean lag stack ``A_1..A_p`` prevailing at date ``t``.

        Args:
            t: Row of the estimation sample, ``0..nobs-1``.

        Returns:
            A ``(p, k, k)`` stack; ``[l - 1, i, j]`` is the posterior mean
            coefficient on lag ``l`` of variable ``j`` in equation ``i`` at
            date ``t``.

        Raises:
            SpecificationError: If ``t`` is out of range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> stack = res.coefficients_at(10)
            >>> stack.shape
            (1, 2, 2)
            >>> bool(np.isclose(stack[0, 1, 0], res.coefficient_path("y2", "y1.L1")[10, 1]))
            True
            >>> res.coefficients_at(res.nobs)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: t must be in 0..98; got 99.
        """
        if not 0 <= t < self.nobs:
            raise SpecificationError(f"t must be in 0..{self.nobs - 1}; got {t}.")
        k, w, d = self.k_endog, self._width, self._n_deterministic
        rows = self.beta_mean[t].reshape(k, w)
        return np.stack([rows[:, d + lag * k : d + (lag + 1) * k] for lag in range(self.order)])

    def stability_path(self) -> npt.NDArray[np.float64]:
        """Largest companion-root modulus of the posterior mean VAR at each date.

        The time-varying analogue of a stability check, and the honest form
        of it: a drifting model can wander through locally explosive
        parameter regions and back, and this path is where that shows.

        Returns:
            An array of length ``nobs``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> roots = res.stability_path()
            >>> roots.shape, bool(np.all((0.2 < roots) & (roots < 0.9)))
            ((99,), True)
            >>> first = np.abs(np.linalg.eigvals(res.coefficients_at(0)[0])).max()
            >>> bool(np.isclose(roots[0], first))
            True
        """
        out = np.empty(self.nobs)
        for t in range(self.nobs):
            eigs = np.linalg.eigvals(companion_matrix(self.coefficients_at(t)))
            out[t] = float(np.abs(eigs).max(initial=0.0))
        return out

    def _drift_note(self) -> str:
        """One line on how much the coefficients actually moved.

        Returns:
            The largest absolute change of any posterior mean coefficient
            between the first and last date, set against the median width
            of the 68% bands, with the reminder that drift inside the band
            is not evidence of time variation.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=1, training=30).fit(n_draws=20, n_burn=8, seed=0)
            >>> res._drift_note().startswith("Posterior mean drift over the sample")
            True
        """
        moved = np.abs(self.beta_mean[-1] - self.beta_mean[0])
        spread = self.beta_high - self.beta_low
        typical = float(np.median(spread))
        return (
            f"Posterior mean drift over the sample: max |beta_T - beta_1| = "
            f"{float(moved.max()):.4f} against a median 68% band width of "
            f"{typical:.4f}; drift smaller than the band is not evidence of "
            "time variation."
        )

    def _family(self) -> str:
        """Specification stem for display.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> TVPVAR(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)._family()
            'TVP-VAR'
        """
        return "TVP-VAR"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with its own first-lag posterior mean at
            the first and last estimation dates; the metadata carries the
            draw bookkeeping, the training split and the state dimension;
            the notes state the drift against the band width, the largest
            local companion root, the training-sample convention and the
            absent likelihood surface.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=1, training=30).fit(n_draws=20, n_burn=8, seed=0)
            >>> table = res._summary_table()
            >>> table.title, table.columns, len(table.rows), len(table.notes)
            ('TVP-VAR(1) Results', ('equation', 'own L1 at start', 'own L1 at end'), 2, 4)
            >>> table.metadata[5], table.metadata[7]
            (('Training', '30'), ('State dimension', '6'))
        """
        roots = self.stability_path()
        rows = tuple(
            (
                name,
                f"{self.coefficient_path(name, f'{name}.L1')[0, 1]:.4f}" if self.order else "-",
                f"{self.coefficient_path(name, f'{name}.L1')[-1, 1]:.4f}" if self.order else "-",
            )
            for name in self.names
        )
        notes = [
            self._drift_note(),
            f"Max |companion root| of the posterior mean path: "
            f"{float(roots.max()):.4f} (a drifting model may pass through "
            "locally explosive dates; stability_path() shows where).",
            "The training sample calibrates the priors and is then discarded; "
            "all paths are indexed on the estimation sample that follows it.",
            "No likelihood, parameter count, or information criteria are "
            "reported: this is a Gibbs posterior and has none.",
        ]
        return SummaryTable(
            title=f"{self._family()}({self.order}) Results",
            metadata=(
                ("Model", f"{self._family()}({self.order})"),
                ("Draws", f"{self.n_draws} ({self.n_burn} burn, thin {self.thin})"),
                ("Variables", f"{self.k_endog}"),
                ("Kept", f"{self.n_kept}"),
                ("Observations", f"{self.nobs}"),
                ("Training", f"{self.training}"),
                ("Trend", self.trend),
                ("State dimension", f"{self.beta_mean.shape[1]}"),
            ),
            columns=("equation", "own L1 at start", "own L1 at end"),
            rows=rows,
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPVARSVResult(TVPVARResult):
    r"""A fitted TVP-VAR with stochastic volatility, after Primiceri (2005).

    Everything the homoskedastic result reports, plus the volatility side:
    the innovation covariance drifts through the triangular factorization

    .. math::

       \Sigma_t = A^{-1} H_t A^{-\top}, \qquad
       H_t = \mathrm{diag}\bigl(e^{h_{1,t}}, \ldots, e^{h_{k,t}}\bigr),
       \qquad h_t = h_{t-1} + \zeta_t, \quad \zeta_t \sim N(0, W),

    with :math:`A` a constant unit-lower-triangular matrix and the log
    variances random walks drawn by the Kim-Shephard-Chib mixture sampler.
    The impact and log-variance draws are retained because the structural
    wrapper needs their posterior, not a point.

    Note:
        ``sigma_u`` on this result is the posterior mean of the
        *end-of-sample* covariance :math:`\Sigma_T`; the full path is in
        ``h_draws`` and ``impact_draws``. Because :math:`A` is triangular
        in the order of ``names``, the recursive identification is fixed
        when the model is fitted: :class:`TimeVaryingSVAR` refuses a
        different ordering on this result.

    Attributes:
        h_draws: ``(S, nobs, k)`` kept log-variance path draws.
        impact_draws: ``(S, k, k)`` kept draws of ``A^{-1}``.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the
            log volatilities.

    See Also:
        * :class:`TVPVARSV` -- the model that produces this record.
        * :class:`TVPVARResult` -- the homoskedastic parent.
        * :class:`~cultivars.multivariate.large_dim.volatility.BVARSVResult`
          -- constant coefficients with the same drifting covariance.

    References:
        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

        Del Negro, M., & Primiceri, G. E. (2015). Time varying structural
        vector autoregressions and monetary policy: A corrigendum. *Review
        of Economic Studies*, 82(4), 1342-1345.

        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> res = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> res.h_draws.shape, res.impact_draws.shape, res.vol_of_vol.shape
        ((13, 99, 2), (13, 2, 2), (2,))
        >>> bool(np.allclose(np.diagonal(res.impact_draws, axis1=1, axis2=2), 1.0))
        True
        >>> bool(np.allclose(res.impact_draws[0], np.tril(res.impact_draws[0])))
        True
        >>> res.volatility_path("y1").shape, res._family()
        ((99, 3), 'TVP-VAR-SV')
    """

    h_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, nobs, k)`` kept log-variance paths, orthogonalized shocks. Kept out of the repr."""
    impact_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k, k)`` kept draws of the unit-lower-triangular ``A^{-1}``. Kept out of the repr."""
    vol_of_vol: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` posterior mean random-walk variances of the log volatilities."""

    def volatility_path(self, name: str) -> npt.NDArray[np.float64]:
        """One equation's posterior innovation standard deviation path.

        The standard deviation of the *orthogonalized* innovation
        ``exp(h_t / 2)`` -- the object the model actually drifts; the
        reduced-form variances mix these through ``A^{-1}``.

        Args:
            name: An endogenous variable.

        Returns:
            An ``(nobs, 3)`` array with columns ``(low, mean, high)``.

        Raises:
            SpecificationError: If the variable is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> path = res.volatility_path("y1")
            >>> path.shape, bool(np.all(path > 0.0))
            ((99, 3), True)
            >>> bool(np.all(path[:, 0] <= path[:, 1])), bool(np.all(path[:, 1] <= path[:, 2]))
            (True, True)
            >>> bool(np.allclose(path[:, 1], np.exp(0.5 * res.h_draws[:, :, 0]).mean(axis=0)))
            True
            >>> res.volatility_path("y3")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown variable 'y3'; expected one of ...
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        i = self.names.index(name)
        sd = np.exp(0.5 * self.h_draws[:, :, i])
        return np.column_stack(
            [
                np.quantile(sd, 0.16, axis=0),
                sd.mean(axis=0),
                np.quantile(sd, 0.84, axis=0),
            ]
        )

    def _family(self) -> str:
        """Specification stem for display.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> TVPVARSV(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)._family()
            'TVP-VAR-SV'
        """
        return "TVP-VAR-SV"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPSVARResult(_SummaryMixin, _ConvergenceMixin):
    r"""A structurally interpreted TVP-VAR: one recursive declaration, a path of answers.

    Holds the fitted reduced form and the posterior mean structural impact
    matrix :math:`B_t` at every estimation date, lower triangular in the
    order of ``names``. For a homoskedastic source :math:`B_t` is the
    Cholesky factor of the posterior mean covariance, the same at every
    date; for a stochastic-volatility source it is the posterior mean of
    :math:`A^{-1} H_t^{1/2}` over the kept draws, so the impact of each
    shock drifts with its volatility while the contemporaneous ordering
    stays fixed. Structural responses at a date are the date's coefficient
    stack propagated and rotated by its impact.

    Note:
        Both the impacts and the responses are posterior-mean plug-ins:
        :math:`B_t` is averaged over draws before it is combined with the
        averaged coefficient path, so a response here is not the posterior
        mean of the response. The source result retains ``beta_draws`` (and
        ``h_draws``, ``impact_draws`` under SV) for a draw-by-draw
        propagation. This record retains no draw arrays of its own, so
        :meth:`convergence` must be read on the source.

    Attributes:
        source: The fitted reduced-form result, homoskedastic or SV.
        impacts: ``(nobs, k, k)`` posterior mean structural impact path --
            constant rows repeated for the homoskedastic model.

    See Also:
        * :class:`TimeVaryingSVAR` -- the declaration that produces this
          record.
        * :class:`TVPVARResult` -- the homoskedastic source.
        * :class:`TVPVARSVResult` -- the stochastic-volatility source.

    References:
        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> source = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> svar = TimeVaryingSVAR(source).identify()
        >>> svar.names, svar.nobs, svar.impacts.shape
        (('y1', 'y2'), 99, (99, 2, 2))
        >>> bool(np.allclose(svar.impact(0), np.linalg.cholesky(source.sigma_u)))
        True
        >>> bool(np.allclose(svar.impact(0), svar.impact(98)))
        True
        >>> svar.irf(4, t=50).shape
        (5, 2, 2)
    """

    source: TVPVARResult = field(repr=False)
    """The fitted reduced form, :class:`TVPVARResult` or :class:`TVPVARSVResult`."""
    impacts: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k, k)`` lower-triangular posterior mean impact path. Kept out of the repr."""

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, in column order; the recursive ordering.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> source = TVPVAR(y, order=1, training=30, names=("r", "pi")).fit(
            ...     n_draws=12, n_burn=4, seed=0
            ... )
            >>> TimeVaryingSVAR(source).identify().names
            ('r', 'pi')
        """
        return self.source.names

    @property
    def nobs(self) -> int:
        """Estimation rows, the length of the impact path.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> source = TVPVAR(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> svar = TimeVaryingSVAR(source).identify()
            >>> svar.nobs == source.nobs == svar.impacts.shape[0]
            True
        """
        return self.source.nobs

    def impact(self, t: int) -> npt.NDArray[np.float64]:
        """The posterior mean structural impact matrix at date ``t``.

        Args:
            t: Row of the estimation sample.

        Returns:
            A lower-triangular ``(k, k)`` matrix.

        Raises:
            SpecificationError: If ``t`` is out of range.

        Example:
            Under stochastic volatility the impact drifts from date to date
            while staying lower triangular:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> source = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> svar = TimeVaryingSVAR(source).identify()
            >>> first, last = svar.impact(0), svar.impact(98)
            >>> bool(np.allclose(first, np.tril(first))), bool(np.any(first != last))
            (True, True)
            >>> svar.impact(99)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: t must be in 0..98; got 99.
        """
        if not 0 <= t < self.nobs:
            raise SpecificationError(f"t must be in 0..{self.nobs - 1}; got {t}.")
        return self.impacts[t]

    def irf(
        self, horizon: int = 20, *, t: int, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Structural impulse responses at date ``t``, held frozen.

        The same linearization every regime family declares, in its drifting
        form: coefficients and impact are frozen at date ``t``'s posterior
        mean, so the response reads "the transmission mechanism prevailing
        at t", exact only while the parameters do not drift over the horizon
        -- and a posterior-mean plug-in besides, with the path uncertainty
        available in the source's retained draws.

        Args:
            horizon: Largest lead to return.
            t: Date whose prevailing dynamics to propagate.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the response of variable ``i`` at lead ``h`` to structural
            shock ``j``, as of date ``t``.

        Raises:
            SpecificationError: If ``horizon`` is negative or ``t`` out of
                range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> source = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> svar = TimeVaryingSVAR(source).identify()
            >>> response = svar.irf(4, t=50)
            >>> response.shape, bool(np.allclose(response[0], svar.impact(50)))
            ((5, 2, 2), True)
            >>> expected = source.coefficients_at(50)[0] @ svar.impact(50)
            >>> bool(np.allclose(response[1], expected))
            True
            >>> bool(np.allclose(svar.irf(4, t=50, cumulative=True)[-1], response.sum(axis=0)))
            True
            >>> svar.irf(-1, t=0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be non-negative; got -1.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        stack = self.source.coefficients_at(t)
        k, p = len(self.names), self.source.order
        psi = np.empty((horizon + 1, k, k))
        if p == 0:
            psi[:] = 0.0
            psi[0] = np.eye(k)
        else:
            selector = np.zeros((k, k * p))
            selector[:, :k] = np.eye(k)
            power = np.eye(k * p)
            companion = companion_matrix(stack)
            for h in range(horizon + 1):
                psi[h] = selector @ power @ selector.T
                power = power @ companion
        out = psi @ self.impact(t)
        return np.cumsum(out, axis=0) if cumulative else out

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with the absolute diagonal impact at the
            first and last dates; the metadata names the scheme, the reduced
            form and the draw count; the notes state where the ordering was
            declared (at estimation under SV, here otherwise), the
            frozen-at-``t`` reading of ``irf``, and the plug-in nature of the
            point summaries.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> source = TVPVAR(y, order=1, training=30).fit(n_draws=20, n_burn=8, seed=0)
            >>> table = TimeVaryingSVAR(source).identify()._summary_table()
            >>> table.title, len(table.rows), len(table.notes)
            ('TVP-SVAR (TVP-VAR) Results', 2, 3)
            >>> table.columns
            ('equation', '|impact| at start', '|impact| at end')
            >>> table.metadata[:2]
            (('Scheme', 'recursive'), ('Reduced form', 'TVP-VAR(1)'))
            >>> table.rows[0][1] == table.rows[0][2]
            True
        """
        first = np.abs(np.diagonal(self.impacts[0]))
        last = np.abs(np.diagonal(self.impacts[-1]))
        rows = tuple(
            (name, f"{first[i]:.4f}", f"{last[i]:.4f}") for i, name in enumerate(self.names)
        )
        sv = isinstance(self.source, TVPVARSVResult)
        notes = [
            "The identifying restriction is recursive in the order of names, "
            + (
                "declared at estimation time through the triangular "
                "factorization: a different ordering is a different fitted "
                "model, not a different rotation of this one."
                if sv
                else "applied to the posterior mean innovation covariance."
            ),
            "irf(t=...) freezes coefficients and impact at date t's posterior "
            "mean: read it as the transmission mechanism prevailing at t, "
            "exact only while parameters do not drift over the horizon.",
            "Point summaries are posterior-mean plug-ins; the source result "
            "retains the draws for full uncertainty propagation.",
        ]
        return SummaryTable(
            title=f"TVP-SVAR ({self.source._family()}) Results",
            metadata=(
                ("Scheme", "recursive"),
                ("Reduced form", f"{self.source._family()}({self.source.order})"),
                ("Observations", f"{self.nobs}"),
                ("Kept draws", f"{self.source.n_kept}"),
            ),
            columns=("equation", "|impact| at start", "|impact| at end"),
            rows=rows,
            notes=tuple(notes),
        )


class TVPVAR(_TimeVaryingVectorAutoRegressionModel[TVPVARResult]):
    r"""Time-varying-parameter VAR with constant innovation covariance.

    The stacked coefficient vector follows a random walk,
    :math:`\beta_t = \beta_{t-1} + \eta_t` with :math:`\eta_t \sim N(0, Q)`,
    and the innovations are homoskedastic, :math:`u_t \sim N(0, \Sigma)`.
    The priors come from a training sample split off the front: its OLS
    estimate :math:`\hat b_0` and covariance :math:`\hat V_0` give the
    initial state :math:`\beta_0 \sim N(\hat b_0, 4 \hat V_0)`, the drift
    covariance an inverse-Wishart prior with scale
    :math:`k_Q^2\,\nu_Q\,\hat V_0` and :math:`\nu_Q = \max(\tau, D + 2)`
    degrees of freedom, and :math:`\Sigma` an inverse-Wishart prior centred
    on the training covariance with :math:`k + 2` degrees of freedom. The
    training rows are then discarded.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _prior: The base slot, unused: the priors here come from the
            training sample.
        _training: The resolved training-sample length.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        training: Rows split off the front to calibrate the priors, then
            discarded. Defaults to roughly forty, floored by what the prior
            regression needs.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the specification is malformed or the
            training sample is shorter than ``w + k + 2`` rows, the floor
            at which the prior regression is identified.
        DimensionError: If fewer than ``2 w`` estimation rows remain after
            the training split.

    See Also:
        * :class:`TVPVARResult` -- the record ``fit`` returns.
        * :class:`TVPVARSV` -- the same drift with stochastic volatility.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          constant-coefficient conjugate model.

    References:
        Cogley, T., & Sargent, T. J. (2005). Drifts and volatilities:
        Monetary policies and outcomes in the post WWII US. *Review of
        Economic Dynamics*, 8(2), 262-302.

        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     a = 0.4 + 0.3 * t / 140
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> res.coefficient_path("y1", "y1.L1").shape
        (99, 3)
        >>> TVPVAR(y, order=1).training, TVPVAR(y, order=2, training=20).training
        (40, 20)
        >>> TVPVAR(y, order=1, training=4)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: a training sample of 4 rows cannot identify ...
    """

    __slots__ = ()

    def fit(
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        k_drift: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> TVPVARResult:
        """Estimate by Gibbs sampling.

        Each iteration draws the whole coefficient path by one
        Durbin-Koopman simulation-smoother pass given ``(Q, Sigma)``, then
        ``Q`` from its inverse-Wishart conditional on the path's
        increments, then ``Sigma`` from its inverse-Wishart conditional on
        the residuals. ``ceil((n_draws - n_burn) / thin)`` paths are kept.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_drift: Primiceri's ``k_Q``: prior scale of coefficient drift
                relative to the training coefficient uncertainty. The single
                most consequential number in the model -- it is what
                disciplines a parameter path with more parameters than data.
                At the default ``0.01`` a slow drift of several tenths over
                a few hundred observations is shrunk to a few hundredths;
                values of ``0.05`` to ``0.3`` let such drift through.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`TVPVARResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If a conditional draw collapses.

        Example:
            The drift prior is the dial: on a coefficient that climbs by
            0.6 over the sample, the posterior mean path moves far more
            under a loose ``k_drift`` than under the default:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((240, 2))
            >>> phi = np.linspace(0.2, 0.8, 240)
            >>> for t in range(1, 240):
            ...     y[t, 0] = phi[t] * y[t - 1, 0] + rng.standard_normal()
            ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.3 * y[t - 1, 0] + rng.standard_normal()
            >>> model = TVPVAR(y, order=1, training=40)
            >>> tight = model.fit(n_draws=40, n_burn=15, seed=0)
            >>> loose = model.fit(n_draws=40, n_burn=15, k_drift=0.3, seed=0)
            >>> moved = lambda r: r.coefficient_path("y1", "y1.L1")[[0, -1], 1]
            >>> bool(np.diff(moved(loose))[0] > 3.0 * np.diff(moved(tight))[0])
            True
            >>> tight.n_kept, tight.n_draws, tight.n_burn, tight.thin
            (13, 40, 15, 2)
            >>> model.fit(n_draws=10, n_burn=10)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_draws (10) must exceed n_burn (10).
        """
        fit = self._fit_tvp(
            sv=False,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            k_drift=k_drift,
            k_vol=0.01,
            seed=seed,
        )
        return TVPVARResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            training=fit.training,
            nobs=fit.nobs,
            beta_mean=fit.beta_mean,
            beta_low=fit.beta_low,
            beta_high=fit.beta_high,
            beta_draws=fit.beta_draws,
            state_cov=fit.state_cov,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )


class TVPVARSV(TVPVAR):
    r"""Time-varying-parameter VAR with stochastic volatility, after Primiceri (2005).

    Coefficients drift as in :class:`TVPVAR`; in addition the innovation
    covariance moves every period through

    .. math::

       \Sigma_t = A^{-1} H_t A^{-\top}, \qquad
       H_t = \mathrm{diag}(e^{h_t}), \qquad
       h_t = h_{t-1} + \zeta_t, \quad \zeta_t \sim N(0, W),

    with a constant unit-lower-triangular :math:`A` and random-walk log
    variances -- the cited workhorse for "has policy changed, or have the
    shocks?", which is unanswerable with either ingredient alone. The
    log-variance block is the Kim-Shephard-Chib mixture sampler; the rows
    of :math:`A` are Gaussian regressions of each orthogonalized residual
    on the ones above it; :math:`W` is diagonal, each entry an
    inverse-Gamma draw with prior scale set by ``k_vol``.

    The triangular factorization is taken in the order of ``names``, and
    that ordering is a structural declaration made at estimation time; see
    :class:`TVPSVAR`.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, in the recursive order, from the base
            model.
        _prior: The base slot, unused: the priors come from the training
            sample.
        _training: The resolved training-sample length.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        training: Rows split off the front to calibrate the priors.
        trend: Deterministic terms.
        names: One label per variable, in the intended recursive order.

    Raises:
        SpecificationError: If the specification is malformed or the
            training sample is below the identification floor.
        DimensionError: If too few estimation rows remain after the split.

    See Also:
        * :class:`TVPVARSVResult` -- the record ``fit`` returns.
        * :class:`TVPVAR` -- the homoskedastic parent.
        * :class:`~cultivars.multivariate.large_dim.volatility.BVARSV` --
          constant coefficients with the same volatility block.

    References:
        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

        Del Negro, M., & Primiceri, G. E. (2015). Time varying structural
        vector autoregressions and monetary policy: A corrigendum. *Review
        of Economic Studies*, 82(4), 1342-1345.

        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.

    Example:
        Innovations whose scale triples over the sample; with the
        volatility prior loosened the posterior mean path ends well above
        where it starts, while the default ``k_vol`` barely lets it move:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     scale = 0.5 + 1.0 * t / 140
        ...     y[t] = 0.5 * y[t - 1] + scale * rng.standard_normal(2)
        >>> res = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> res.volatility_path("y1").shape
        (99, 3)
        >>> model = TVPVARSV(y, order=1, training=40)
        >>> loose = model.fit(n_draws=40, n_burn=15, k_vol=0.1, seed=0).volatility_path("y1")[:, 1]
        >>> tight = res.volatility_path("y1")[:, 1]
        >>> bool(loose[-1] / loose[0] > 2.0), bool(tight[-1] / tight[0] < 1.3)
        (True, True)
    """

    __slots__ = ()

    def fit(  # type: ignore[override]
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        k_drift: float = 0.01,
        k_vol: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> TVPVARSVResult:
        """Estimate by Gibbs sampling with the KSC volatility block.

        Each iteration draws the coefficient path given the current
        covariance path, ``Q`` given the increments, and then the
        volatility block -- the log-variance paths through the KSC mixture
        with the simulation smoother, the rows of ``A``, and the diagonal
        ``W`` -- given the residuals.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_drift: Primiceri's ``k_Q``: prior scale of coefficient drift.
            k_vol: Primiceri's ``k_W``: prior scale of the log-volatility
                random walk.
            seed: Seed or generator.

        Returns:
            The fitted :class:`TVPVARSVResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If a conditional draw collapses.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((140, 2))
            >>> for t in range(1, 140):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> res = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
            >>> res.n_kept, res.h_draws.shape, res.impact_draws.shape
            (13, (13, 99, 2), (13, 2, 2))
            >>> isinstance(res, TVPVARResult), type(res).__name__
            (True, 'TVPVARSVResult')
        """
        fit = self._fit_tvp(
            sv=True,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            k_drift=k_drift,
            k_vol=k_vol,
            seed=seed,
        )
        assert fit.h_draws is not None
        assert fit.impact_draws is not None
        assert fit.vol_of_vol is not None
        return TVPVARSVResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            training=fit.training,
            nobs=fit.nobs,
            beta_mean=fit.beta_mean,
            beta_low=fit.beta_low,
            beta_high=fit.beta_high,
            beta_draws=fit.beta_draws,
            state_cov=fit.state_cov,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
            h_draws=fit.h_draws,
            impact_draws=fit.impact_draws,
            vol_of_vol=fit.vol_of_vol,
        )


class TVPSVAR:
    """Recursive structural interpretation of a fitted TVP-VAR, per date.

    Not an ``_IdentificationModel``: a drifting system is a continuum of
    closed systems, one per date, and what this class packages is the path
    of identifications under a single recursive declaration.

    The declaration works differently across the two reduced forms, and the
    difference is enforced rather than papered over. For the homoskedastic
    model the ordering is applied here, to the posterior mean covariance,
    and any permutation of ``names`` is admissible. For the SV model the
    triangular factorization *was the estimation*: ``A`` and the volatility
    paths were drawn in the order of ``names``, so the ordering was declared
    when the model was fitted, and asking for a different one here is asking
    for a different fitted model -- refit with the columns reordered.

    Attributes:
        _source: The fitted reduced-form result.
        _order: The requested recursive ordering, or ``None`` for the order
            of ``names``.

    Args:
        source: A fitted :class:`TVPVARResult` or :class:`TVPVARSVResult`.
        order: Recursive ordering for the *homoskedastic* model; ``None``
            keeps the order of ``names``. Refused for an SV source.

    Raises:
        SpecificationError: If the source is not a fitted TVP result, an
            ordering is requested for an SV source, or the ordering is not a
            permutation of ``names``.

    See Also:
        * :class:`TVPSVARResult` -- the record ``identify`` returns.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.Recursive`
          -- the same declaration on a constant-coefficient system.

    References:
        Primiceri, G. E. (2005). Time varying structural vector
        autoregressions and monetary policy. *Review of Economic Studies*,
        72(3), 821-852.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> svar = TVPSVAR(res).identify()
        >>> svar.irf(4, t=50).shape
        (5, 2, 2)
        >>> TVPSVAR(y)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: TimeVaryingSVAR constructs with a fitted ...
        >>> TVPSVAR(res, order=("y2", "y3"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: order must be a permutation of ('y1', 'y2'); ...
    """

    __slots__ = ("_order", "_source")

    def __init__(
        self,
        source: TVPVARResult,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the source and the ordering request.

        Args:
            source: A fitted :class:`TVPVARResult` or
                :class:`TVPVARSVResult`.
            order: Recursive ordering for a homoskedastic source, or
                ``None``.

        Raises:
            SpecificationError: If the source is not a TVP result, an
                ordering is given for an SV source, or the ordering is not
                a permutation of ``names``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> plain = TVPVAR(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> TVPSVAR(plain, order=("y2", "y1"))._order
            ('y2', 'y1')
            >>> sv = TVPVARSV(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> TVPSVAR(sv)._order is None
            True
            >>> TVPSVAR(sv, order=("y2", "y1"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: the SV model's recursive ordering is ...
        """
        if not isinstance(source, TVPVARResult):
            raise SpecificationError(
                "TimeVaryingSVAR constructs with a fitted TVPVAR or TVPVARSV "
                f"result; got {type(source).__name__}."
            )
        if order is not None and isinstance(source, TVPVARSVResult):
            raise SpecificationError(
                "the SV model's recursive ordering is declared at estimation "
                "time through the triangular factorization; a different "
                "ordering is a different fitted model. Refit TVPVARSV with "
                "the columns in the ordering you intend."
            )
        if order is not None:
            resolved = tuple(str(name) for name in order)
            if sorted(resolved) != sorted(source.names):
                raise SpecificationError(
                    f"order must be a permutation of {source.names}; got {resolved}."
                )
            self._order: tuple[str, ...] | None = resolved
        else:
            self._order = None
        self._source = source

    def identify(self) -> TVPSVARResult:
        """Assemble the impact path under the recursive declaration.

        Returns:
            The :class:`TVPSVARResult`. For an SV source, the impact at each
            date is the posterior mean of ``A^{-1} diag(exp(h_t / 2))``
            across the kept draws; for a homoskedastic source it is the
            Cholesky factor of the posterior mean covariance, repeated.

        Example:
            A reordered homoskedastic identification is the Cholesky factor
            in the new order, written back into the original positions, so
            it still reproduces the covariance:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = TVPVAR(y, order=1, training=30).fit(n_draws=12, n_burn=4, seed=0)
            >>> plain = TVPSVAR(res).identify().impact(0)
            >>> flipped = TVPSVAR(res, order=("y2", "y1")).identify().impact(0)
            >>> bool(np.allclose(plain @ plain.T, res.sigma_u))
            True
            >>> bool(np.allclose(flipped @ flipped.T, res.sigma_u))
            True
            >>> bool(np.allclose(plain, flipped))
            False
            >>> bool(np.isclose(flipped[0, 1], 0.0)) or bool(np.isclose(flipped[1, 0], 0.0))
            True
        """
        source = self._source
        n, k = source.nobs, source.k_endog
        if isinstance(source, TVPVARSVResult):
            scale = np.exp(0.5 * source.h_draws)
            impacts = np.einsum("sij,stj->stij", source.impact_draws, scale).mean(axis=0)
        else:
            sigma = source.sigma_u
            if self._order is not None:
                perm = [source.names.index(name) for name in self._order]
                inverse = np.argsort(perm)
                reordered = sigma[np.ix_(perm, perm)]
                chol = np.linalg.cholesky(reordered)
                chol = chol[np.ix_(inverse, inverse)]
            else:
                chol = np.linalg.cholesky(sigma)
            impacts = np.broadcast_to(chol, (n, k, k)).copy()
        return TVPSVARResult(source=source, impacts=impacts)
