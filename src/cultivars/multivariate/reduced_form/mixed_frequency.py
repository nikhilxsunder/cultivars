# filepath: /src/cultivars/multivariate/reduced_form/mixed_frequency.py
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
r"""Mixed-frequency vector autoregressions: two models for two different problems.

Both models here relate series sampled at different frequencies, and that is
the whole of their resemblance. They differ in which frequency the *model*
lives at, and the difference decides everything else -- estimator, result
surface, and whether a maximum-likelihood ``fit`` is defensible at all.

:class:`MFVAR` puts the model at the **high** frequency,

.. math::

   z_t = c + \sum_{i=1}^{p} A_i\, z_{t-i} + u_t,
   \qquad
   y^{\text{obs}}_{it} = \sum_{j=0}^{m-1} w_j\, z_{i,t-j}
   \;\;\text{on the dates a reading lands,}

so the monthly path is latent, a quarterly reading is a linear functional of
it, and the Kalman smoother recovers the path. This is the Mariano-Murasawa
state-space formulation. Its one structural elegance is that the observation
matrix is *constant*: with the state stacking :math:`\max(p, m)` lags, the
calendar is not in :math:`Z`, it is in the missing-value pattern of the
sample, and the filter already masks missing rows. Nothing about the model
changes between a month in which the quarterly series is observed and one in
which it is not.

:class:`MIDASVAR` puts the model at the **low** frequency,

.. math::

   y_t = D d_t + \sum_{i=1}^{p} A_i\, y_{t-i} + B\, x_t(\theta) + u_t,
   \qquad
   x_t(\theta)_q = \sum_{j=1}^{L} w_j(\theta_q)\, x^{(q)}_{t,j},
   \quad w_j \propto e^{\theta_1 j + \theta_2 j^2},

so each high-frequency regressor is compressed into one column per
low-frequency date by a lag polynomial with two parameters, and a year of
daily history costs two degrees of freedom rather than two hundred and fifty.
This is Ghysels-Santa-Clara-Valkanov.

Two commitments shape the surface. First, the asymmetry in what the two
classes expose is deliberate and is documented at each class.
:class:`MIDASVAR` estimates, by profiling the polynomial out of a Gaussian
likelihood that identifies everything it touches. :class:`MFVAR` does not,
and the reason is recorded in its docstring rather than buried in a release
note: with a temporally aggregated variable, the Gaussian likelihood is not
merely flat in that variable's own innovation variance, it is *maximized away
from the truth*, and the error grows with the sample. A model that returns a
confident wrong number is worse than one that declines, so :class:`MFVAR`
takes its parameters -- from a VAR fitted on whatever high-frequency sample
exists, through ``from_result`` -- and filters with them. Second, both
results are honest about what is conditional. The MIDAS standard errors
condition on the estimated polynomial and say so, with ``joint_stderr()`` as
the full-information alternative; both forecasts condition on a supplied
future high-frequency path and refuse to invent one; and the MFVAR record,
which carries the whole propagation surface because the smoothed path is
complete, labels its coefficients as supplied rather than estimated.

Layout. :class:`MFVAR` validates its parameters and calendar in place,
builds the stacked system through ``_mixed_frequency_system`` in
``_core._converters`` with the per-kind weights of ``_aggregation_weights``,
and runs the filter and the Durbin-Koopman smoother of
``_LinearGaussianStateSpace`` from the state-space layer; ``smooth()`` packs
the path into :class:`MFVARResult`, which takes its summary from
``_SummaryMixin`` and its propagation surface from
``_VectorPropagationMixin``. :class:`MIDASVAR` extends
``_VectorAutoRegressionModel`` in ``_internals``, stacks the sub-period
windows with ``_midas_windows`` and the weight curves with
``_midas_weights`` from ``_core._polynomials``, hands the profile problem to
``_MidasProfileObjective`` and ``_maximize_likelihood``, and reuses the
base's Gaussian moments for everything conditional on ``theta``;
:class:`MIDASVARResult` is a full ``_VectorResult`` with the summary,
comparison, inference and propagation mixins. The closed-system estimator
both lean on is
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`; the
filtering substrate is :mod:`~cultivars.state_space`; the Bayesian route to
an *estimated* mixed-frequency VAR belongs to :mod:`~cultivars.bayes`.

References:
    Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business
    cycles based on monthly and quarterly series. *Journal of Applied
    Econometrics*, 18(4), 427-443.

    Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch:
    Mixed data sampling regression models. CIRANO Working Paper 2004s-20.

    Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS regressions: Further
    results and new directions. *Econometric Reviews*, 26(1), 53-90.

    Ghysels, E. (2016). Macroeconomics and the reality of mixed frequency
    data. *Journal of Econometrics*, 193(2), 294-314.

    Schorfheide, F., & Song, D. (2015). Real-time forecasting with a
    mixed-frequency VAR. *Journal of Business & Economic Statistics*, 33(3),
    366-380.

Example:
    One monthly indicator, two quarterly questions. A monthly pair shares a
    VAR(1) whose second variable is read only as a quarterly average:
    :class:`MFVAR`, given the parameters, infers the missing months. A
    quarterly spending series loads on the indicator's three months with
    weights rising toward the quarter's end: :class:`MIDASVAR` estimates the
    curve from the quarterly sample and the full monthly history:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = np.array([[[0.5, 0.2], [0.3, 0.6]]])
    >>> z = np.zeros((240, 2))
    >>> for t in range(1, 240):
    ...     z[t] = A[0] @ z[t - 1] + rng.standard_normal(2)
    >>> monthly = z.copy()
    >>> monthly[:, 1] = np.nan
    >>> quarters = np.arange(2, 240, 3)
    >>> monthly[quarters, 1] = [z[t - 2 : t + 1, 1].mean() for t in quarters]
    >>> latent = MFVAR(A, np.eye(2), kinds=["high", "flow"], period=3).smooth(monthly)
    >>> error = latent.endog[:, 1] - z[:, 1]
    >>> bool(np.sqrt(np.mean(error**2)) < 0.5 * z[:, 1].std())
    True
    >>> months = z[:, 0].reshape(80, 3)
    >>> spending = months @ [0.2, 0.3, 0.5] + 0.2 * rng.standard_normal(80)
    >>> quarterly = np.column_stack([spending, monthly[quarters, 1]])
    >>> names, exog_names = ["spend", "gdp"], ["ip"]
    >>> midas = MIDASVAR(quarterly, z[:, :1], order=1, period=3, names=names, exog_names=exog_names)
    >>> res = midas.fit()
    >>> res.midas_weights.ravel().round(2), res.midas_coefficients.round(1).ravel()
    (array([0.47, 0.27, 0.25]), array([1. , 0.5]))
    >>> res.forecast(1, exog_high_future=np.zeros((3, 1))).shape
    (1, 2)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from ...engine._core import (
    _AGGREGATION_NOTE,
    _CHOLESKY_NOTE,
    _MIDAS_CONDITIONAL_NOTE,
    _UNSTABLE_NOTE,
    Frequency,
    SummaryTable,
    Trend,
    _aggregation_weights,
    _midas_weights,
    _midas_windows,
    _mixed_frequency_system,
    _validate_observed,
    deterministic_columns,
    lag_matrix,
    validate_exog_matrix,
)
from ...engine._internals import (
    _ComparisonMixin,
    _DurbinKoopmanSmootherResult,
    _KalmanFilterResult,
    _LinearGaussianStateSpace,
    _maximize_likelihood,
    _MidasProfileObjective,
    _SummaryMixin,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["MFVAR", "MIDASVAR", "MFVARResult", "MIDASVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MFVARResult(_SummaryMixin, _VectorPropagationMixin):
    r"""The latent high-frequency path implied by a mixed-frequency sample.

    The model lives at the high frequency,

    .. math::

       z_t = c + \sum_{i=1}^{p} A_i\, z_{t-i} + u_t, \qquad u_t \sim N(0, \Sigma_u),

    and a variable of kind ``"high"`` is observed as :math:`z_{it}` every
    period, a ``"stock"`` as :math:`z_{it}` at the end of each low-frequency
    period, and a ``"flow"`` as the weighted sum
    :math:`\sum_{j=0}^{m-1} w_j z_{i,t-j}` over the :math:`m` sub-periods
    ending at :math:`t`; every other entry of the sample is ``nan``. With
    :math:`(A, \Sigma_u, c)` taken as given, the Kalman smoother over a state
    stacking :math:`\max(p, m)` lags returns the conditional mean and
    covariance of the whole latent path given every reading in the sample,
    and this record is that path dressed as a VAR result.

    Note:
        Nothing here was estimated. :attr:`coefficients`, :attr:`sigma_u` and
        :attr:`deterministic` are the inputs to :class:`MFVAR` echoed back,
        so there are no standard errors, no information criteria and no
        comparison surface, and :attr:`loglikelihood` is the likelihood of
        the observed entries *at those parameters* -- a diagnostic for
        comparing parameter sets, not a fit statistic. What the record does
        carry is the full propagation surface, because the smoothed path is
        complete: forecasts, impulse responses and decompositions read
        :attr:`endog` and :attr:`resid` exactly as a VAR result reads its
        data, with the caveat that :attr:`resid` are innovations of the
        *smoothed* path, smoother than the true innovations where a variable
        was not observed.

    Attributes:
        observed: The mixed-frequency sample as supplied, ``nan`` preserved.
        endog: The inferred latent path, ``(nobs, k)`` and complete. Named
            ``endog`` rather than ``smoothed_state`` on purpose: it is what the
            propagation surface consumes, so forecasting and impulse responses
            work here unchanged, reading the inferred path exactly as a VAR
            result reads its data.
        latent_cov: Smoothed state covariances of the contemporaneous block,
            ``(nobs, k, k)``.
        coefficients: The ``(order, k, k)`` coefficients used, as supplied.
        deterministic: Deterministic coefficients, one row per term; the
            supplied intercept, or zero rows when there was none.
        sigma_u: The ``(k, k)`` innovation covariance used, as supplied.
        resid: Innovations of the inferred path over the effective sample:
            ``z_t - c - A_1 z_{t-1} - ... - A_p z_{t-p}`` evaluated on the
            smoothed path. This is what the historical decomposition and the
            structural shocks consume.
        loglikelihood: Log-likelihood of the observed entries.
        kinds: The frequency role of each variable.
        weights: Explicit per-variable sub-period weights, when supplied.
        period: Sub-periods per low-frequency period.
        names: Variable labels.
        order: Autoregressive order.
        trend: Deterministic specification implied by the intercept.
        nobs: Effective sample size, the rows of :attr:`resid`.

    See Also:
        * :class:`MFVAR` -- the model whose ``smooth()`` returns this record
          and whose docstring records why it does not estimate.
        * :class:`MIDASVARResult` -- the low-frequency alternative, which is
          estimated.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the estimated high-frequency result whose parameters
          ``MFVAR.from_result`` carries here.

    References:
        Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of
        business cycles based on monthly and quarterly series. *Journal of
        Applied Econometrics*, 18(4), 427-443.

        Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
        Space Methods* (2nd ed.). Oxford University Press.

    Example:
        A monthly bivariate VAR(1) in which the second variable is seen only
        as a quarterly average. The smoothed path recovers it to about half
        its standard deviation, the pointwise uncertainty is lowest in the
        middle month of each quarter, where both neighbours are tied to the
        same reading, the implied quarterly readings reproduce the ones
        observed, and the propagation surface runs on the inferred path:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[[0.5, 0.2], [0.1, 0.6]]])
        >>> S = np.array([[1.0, 0.3], [0.3, 1.0]])
        >>> z = np.zeros((240, 2))
        >>> for t in range(1, 240):
        ...     z[t] = A[0] @ z[t - 1] + rng.multivariate_normal(np.zeros(2), S)
        >>> y = z.copy()
        >>> y[:, 1] = np.nan
        >>> for t in range(2, 240, 3):
        ...     y[t, 1] = z[t - 2 : t + 1, 1].mean()
        >>> res = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y)
        >>> res.endog.shape, res.nobs, res.observed_fraction.round(3)
        ((240, 2), 239, array([1.   , 0.333]))
        >>> error = res.endog[:, 1] - z[:, 1]
        >>> bool(np.sqrt(np.mean(error**2)) < 0.5 * z[:, 1].std())
        True
        >>> se = res.latent_stderr[:, 1]
        >>> bool(se[4] < se[3]) and bool(se[4] < se[5])
        True
        >>> implied = res.implied_low_frequency()
        >>> bool(np.allclose(implied[2::3, 1], y[2::3, 1]))
        True
        >>> res.forecast(2).shape, res.irf(4).shape, res.is_stable
        ((2, 2), (5, 2, 2), True)
    """

    observed: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, k)`` sample as supplied, ``nan`` where unobserved. Kept out of the repr."""

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, k)`` smoothed latent path, complete. Kept out of the repr."""

    latent_cov: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, k, k)`` smoothed covariances of the current latent vector.

    Kept out of the repr.
    """

    coefficients: npt.NDArray[np.float64] = field(repr=False)
    """The ``(order, k, k)`` autoregressive matrices, as supplied. Kept out of the repr."""

    deterministic: npt.NDArray[np.float64] = field(repr=False)
    """The ``(1, k)`` intercept row, or ``(0, k)`` when none was supplied. Kept out of the repr."""

    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """The ``(k, k)`` innovation covariance, as supplied and symmetrized. Kept out of the repr."""

    resid: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs - order, k)`` innovations of the smoothed path. Kept out of the repr."""

    loglikelihood: float
    """Gaussian log-likelihood of the observed entries at the supplied parameters."""

    kinds: tuple[Frequency, ...]
    """The frequency role of each variable: ``"high"``, ``"stock"`` or ``"flow"``."""

    period: int
    """High-frequency sub-periods per low-frequency period."""

    names: tuple[str, ...]
    """Variable labels."""

    order: int
    """Latent autoregressive order."""

    trend: str
    """``"c"`` when an intercept was supplied, ``"n"`` otherwise."""

    nobs: int
    """Effective sample size: the rows of :attr:`resid`, ``periods - order``."""

    weights: tuple[npt.NDArray[np.float64] | None, ...] | None = field(default=None, repr=False)
    """Per-variable explicit sub-period weights, most recent first, or ``None``.

    Kept out of the repr.
    """

    @property
    def k_endog(self) -> int:
        """Number of variables.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y).k_endog
            2
        """
        return len(self.names)

    @property
    def latent_stderr(self) -> npt.NDArray[np.float64]:
        """Pointwise standard errors of the latent path, ``(nobs, k)``.

        The diagonal of a smoothed covariance reaches zero from below by a few
        units in the last place wherever a variable is directly observed, and
        an unguarded square root would return ``nan`` for exactly the entries
        the model knows best. The clip is not cosmetic.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> se = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y).latent_stderr
            >>> se.shape, bool(se[:, 0].max() < 1e-6), bool(se[:, 1].min() > 0.1)
            ((24, 2), True, True)
        """
        var = np.diagonal(self.latent_cov, axis1=1, axis2=2)
        return np.sqrt(np.clip(var, 0.0, None))

    @property
    def observed_fraction(self) -> npt.NDArray[np.float64]:
        """Share of periods in which each variable was directly observed.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y).observed_fraction
            array([1.        , 0.33333333])
        """
        return np.isfinite(self.observed).mean(axis=0)

    def implied_low_frequency(self) -> npt.NDArray[np.float64]:
        """Aggregate the inferred latent path back to the observed frequency.

        Returns:
            An ``(nobs, k)`` array whose entries are the model's fitted value
            for a low-frequency reading ending at that period, ``nan`` in the
            first ``period - 1`` rows where no full window exists. Comparing
            this with :attr:`observed` at the periods where a reading exists
            is the residual check for a mixed-frequency model; with no
            measurement noise in the observation equation the two agree to
            rounding by construction, so the informative comparison is at the
            periods *between* readings, against any external benchmark.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> res = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y)
            >>> implied = res.implied_low_frequency()
            >>> bool(np.isnan(implied[:2]).all()), bool(np.allclose(implied[2::3, 1], y[2::3, 1]))
            (True, True)
            >>> bool(np.allclose(implied[2:, 0], res.endog[2:, 0]))
            True
        """
        nobs, size = self.endog.shape
        out = np.full((nobs, size), np.nan, dtype=np.float64)
        for i, kind in enumerate(self.kinds):
            supplied = None if self.weights is None else self.weights[i]
            w = _aggregation_weights(kind, self.period, weights=supplied)
            for t in range(self.period - 1, nobs):
                window = self.endog[t - self.period + 1 : t + 1, i][::-1]
                out[t, i] = float(w @ window)
        return out

    def nowcast(self, index: int = -1) -> dict[str, float]:
        """The latent value of every variable at one period.

        Args:
            index: Position in the sample; negative indexes from the end.

        Returns:
            Mapping from variable name to its inferred value.

        Raises:
            DimensionError: If ``index`` is out of range.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> res = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y)
            >>> current = res.nowcast()
            >>> sorted(current), bool(abs(current["y1"] - y[-1, 0]) < 1e-6)
            (['y1', 'y2'], True)
            >>> res.nowcast(24)
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: index 24 is out of range for 24 periods.
        """
        nobs = self.endog.shape[0]
        pos = index + nobs if index < 0 else index
        if not 0 <= pos < nobs:
            raise DimensionError(f"index {index} is out of range for {nobs} periods.")
        return {name: float(self.endog[pos, i]) for i, name in enumerate(self.names)}

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        One row per variable with its frequency role and observed share; no
        coefficient table, because nothing was estimated. The notes carry
        the stability verdict of the supplied coefficients, the aggregation
        note pointing at :class:`MFVAR` for why they were not estimated, and
        the Cholesky ordering; an unstable system is flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> table = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y)._summary_table()
            >>> table.title, table.rows
            ('MFVAR(1) Inference', (('y1', 'high', '100.0%'), ('y2', 'flow', '33.3%')))
            >>> dict(table.metadata)["Sub-periods"], dict(table.metadata)["Trend"]
            ('3', 'n')
        """
        stability = self.stability_check()
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            _AGGREGATION_NOTE,
            _CHOLESKY_NOTE,
        ]
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        rows = tuple(
            (name, str(self.kinds[i]), f"{float(self.observed_fraction[i]):.1%}")
            for i, name in enumerate(self.names)
        )
        return SummaryTable(
            title=f"MFVAR({self.order}) Inference",
            metadata=(
                ("Model", f"MFVAR({self.order})"),
                ("Log-likelihood", f"{self.loglikelihood:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("Sub-periods", f"{self.period}"),
                ("Periods", f"{self.endog.shape[0]}"),
                ("Trend", self.trend),
            ),
            columns=("variable", "role", "observed"),
            rows=rows,
            notes=tuple(notes),
        )


class MFVAR:
    r"""A latent high-frequency VAR observed through temporal aggregation.

    The model is :math:`z_t = c + A_1 z_{t-1} + \dots + A_p z_{t-p} + u_t` at
    the high frequency; what is observed is :math:`z` itself for some
    variables and a weighted sub-period sum for others, on a calendar
    expressed as ``nan`` in the sample rather than as a time-varying
    observation matrix. The state stacks :math:`\max(p, m)` lags of
    :math:`z_t`, deep enough that a reading spanning :math:`m` sub-periods is
    a fixed linear function of the current state, so the observation matrix
    is constant and the filter's missing-value masking does the rest.

    **This class does not estimate.** The omission is a finding, not an
    unfinished feature. Simulating a three-variable monthly VAR with one
    variable observed only as a quarterly average and estimating by EM:

    =========  ===========  ===========  ===========
    ``T``      ``|A err|``  ``Sigma11``  ``|S err|``
    =========  ===========  ===========  ===========
    control    0.051        1.100        0.060
    1500       0.129        2.629        1.469
    6000       0.183        3.104        1.944
    =========  ===========  ===========  ===========

    against a truth of ``Sigma11 = 1.160``. The control row is the identical
    estimator with nothing aggregated, so the machinery is sound. The error in
    the aggregated rows *grows* with the sample: this is convergence to a wrong
    limit, not sampling noise, and the likelihood confirms it -- the wrong
    answer scores about a hundred log-points above the truth. The mechanism is
    that a quarterly reading constrains a three-month average, and a high
    monthly innovation variance with weaker persistence is nearly
    observationally equivalent to a low one with stronger persistence.

    Priors do not rescue it. A Minnesota prior shrinks ``A`` and cannot reach
    the unidentified direction, which lives in ``Sigma``. An inverse-Wishart
    prior on ``Sigma`` anchored at an aggregation-implied scale fails for a
    sharper reason: the correct anchor requires the high-frequency persistence,
    which is precisely what is not identified. Anchoring even at the true value
    needs a prior worth half the sample to recover the cell.

    The route to an estimated mixed-frequency VAR is Gibbs sampling over
    ``(A, Sigma)`` jointly with a proper prior -- integrating rather than
    maximizing, as Schorfheide and Song do -- which belongs to
    :mod:`cultivars.bayes` once a sampling backend exists. Until then, supply
    parameters and filter: :meth:`from_result` takes them from a VAR fitted on
    whatever high-frequency sample exists, :meth:`smooth` infers the latent
    path, :meth:`nowcast` reads off its last period.

    Attributes:
        _coefficients: The validated ``(order, k, k)`` autoregressive stack.
        _sigma_u: The validated, symmetrized ``(k, k)`` innovation covariance.
        _intercept: The ``(k,)`` latent intercept, or ``None``.
        _kinds: One frequency role per variable.
        _period: Sub-periods per low-frequency period.
        _weights: Per-variable explicit sub-period weights, or ``None``.
        _names: Variable labels.

    See Also:
        * :class:`MFVARResult` -- the record ``smooth()`` returns.
        * :class:`MIDASVAR` -- the low-frequency model, which estimates.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the estimator whose parameters this class borrows, and the right
          model when every variable is observed at the high frequency.
        * :mod:`~cultivars.state_space` -- the Kalman filter and simulation
          smoother the inference runs on.

    References:
        Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of
        business cycles based on monthly and quarterly series. *Journal of
        Applied Econometrics*, 18(4), 427-443.

        Schorfheide, F., & Song, D. (2015). Real-time forecasting with a
        mixed-frequency VAR. *Journal of Business & Economic Statistics*,
        33(3), 366-380.

        Ghysels, E. (2016). Macroeconomics and the reality of mixed frequency
        data. *Journal of Econometrics*, 193(2), 294-314.

    Example:
        >>> import numpy as np
        >>> A = np.array([[[0.5, 0.1], [0.0, 0.4]]])
        >>> S = np.eye(2)
        >>> y = np.full((24, 2), np.nan)
        >>> rng = np.random.default_rng(0)
        >>> y[:, 0] = rng.standard_normal(24)
        >>> y[2::3, 1] = rng.standard_normal(8)
        >>> res = MFVAR(A, S, kinds=["high", "flow"], period=3).smooth(y)
        >>> res.endog.shape
        (24, 2)
    """

    __slots__ = (
        "_coefficients",
        "_intercept",
        "_kinds",
        "_names",
        "_period",
        "_sigma_u",
        "_weights",
    )

    def __init__(
        self,
        coefficients: npt.ArrayLike,
        sigma_u: npt.ArrayLike,
        *,
        kinds: Sequence[Frequency],
        period: int,
        intercept: npt.ArrayLike | None = None,
        weights: Sequence[npt.ArrayLike | None] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the supplied parameters and the aggregation calendar.

        Args:
            coefficients: ``(order, k, k)`` latent autoregressive coefficients,
                or ``(k, k)`` for an order of one.
            sigma_u: ``(k, k)`` latent innovation covariance.
            kinds: One :data:`Frequency` per variable.
            period: High-frequency sub-periods per low-frequency period.
            intercept: Optional ``(k,)`` latent intercept.
            weights: Optional per-variable sub-period weights, most recent
                first; ``None`` entries take the default for the variable's
                kind.
            names: Variable labels. Defaults to ``y1 ... yk``.

        Raises:
            DimensionError: If the coefficient, covariance or intercept shapes
                disagree.
            NumericalError: If the coefficients or covariance are non-finite.
            SpecificationError: If the covariance is asymmetric or indefinite,
                ``period`` is not a positive integer, ``kinds``, ``weights`` or
                ``names`` have the wrong length, a kind is unrecognized, or
                every variable is marked ``high`` -- in which case there is no
                mixed frequency and a plain
                :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
                is the right model.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[0.5, 0.1], [0.0, 0.4]]), np.eye(2)
            >>> model = MFVAR(A, S, kinds=["high", "stock"], period=3, names=["ip", "gdp"])
            >>> model.order, model.k_endog
            (1, 2)
            >>> MFVAR(A, S, kinds=["high", "high"], period=3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: every variable is marked 'high', so nothing ...
            >>> MFVAR(A, np.eye(3), kinds=["high", "flow"], period=3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: sigma_u must have shape (2, 2) to match ...
            >>> MFVAR(A, S, kinds=["high", "daily"], period=3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unrecognized frequency roles ['daily']; ...
        """
        coef = np.asarray(coefficients, dtype=np.float64)
        if coef.ndim == 2:
            coef = coef[np.newaxis]
        if coef.ndim != 3 or coef.shape[1] != coef.shape[2]:
            raise DimensionError(f"coefficients must have shape (order, k, k); got {coef.shape}.")
        if not np.all(np.isfinite(coef)):
            raise NumericalError("coefficients must be finite.")
        size = int(coef.shape[1])

        sig = np.asarray(sigma_u, dtype=np.float64)
        if sig.shape != (size, size):
            raise DimensionError(
                f"sigma_u must have shape ({size}, {size}) to match coefficients; got {sig.shape}."
            )
        if not np.all(np.isfinite(sig)):
            raise NumericalError("sigma_u must be finite.")
        if not np.allclose(sig, sig.T, atol=1e-10):
            raise SpecificationError("sigma_u must be symmetric.")
        if float(np.min(np.linalg.eigvalsh((sig + sig.T) / 2.0))) < -1e-10:
            raise SpecificationError("sigma_u must be positive semidefinite.")

        if int(period) != period or period < 1:
            raise SpecificationError(f"period must be an integer >= 1; got {period!r}.")
        if len(kinds) != size:
            raise SpecificationError(
                f"kinds must have one entry per variable ({size}); got {len(kinds)}."
            )
        allowed = {"high", "stock", "flow"}
        bad = [k for k in kinds if k not in allowed]
        if bad:
            raise SpecificationError(
                f"unrecognized frequency roles {bad}; expected one of {sorted(allowed)}."
            )
        if all(k == "high" for k in kinds):
            raise SpecificationError(
                "every variable is marked 'high', so nothing is aggregated. Use "
                "VAR, which estimates its parameters rather than taking them."
            )
        if weights is not None and len(weights) != size:
            raise SpecificationError(
                f"weights must have one entry per variable ({size}); got {len(weights)}."
            )

        self._coefficients = coef
        self._sigma_u = (sig + sig.T) / 2.0
        self._kinds = tuple(kinds)
        self._period = int(period)
        self._weights = None if weights is None else tuple(weights)
        self._intercept = (
            None if intercept is None else np.asarray(intercept, dtype=np.float64).ravel()
        )
        if self._intercept is not None and self._intercept.shape != (size,):
            raise DimensionError(
                f"intercept must have shape ({size},); got {self._intercept.shape}."
            )
        self._names = tuple(f"y{i + 1}" for i in range(size)) if names is None else tuple(names)
        if len(self._names) != size:
            raise SpecificationError(
                f"names must have one entry per variable ({size}); got {len(self._names)}."
            )

    @classmethod
    def from_result(
        cls,
        result: object,
        *,
        kinds: Sequence[Frequency],
        period: int,
        weights: Sequence[npt.ArrayLike | None] | None = None,
    ) -> Self:
        """Build a filter from an already-estimated high-frequency VAR.

        The intended workflow. Estimate the VAR on whatever high-frequency
        sample exists -- a shorter span, a related vintage, a subset of the
        variables -- then bring those parameters here to infer the latent path
        over the mixed-frequency sample.

        Args:
            result: Any fitted result exposing ``coefficients``, ``sigma_u``,
                and ``names``.
            kinds: One :data:`Frequency` per variable.
            period: High-frequency sub-periods per low-frequency period.
            weights: Optional per-variable sub-period weights.

        Returns:
            A configured :class:`MFVAR`.

        Raises:
            SpecificationError: If ``result`` lacks the required attributes.

        Note:
            The result's ``deterministic`` block is not carried over, so the
            filter runs without an intercept even when the VAR was fitted
            with one; pass ``intercept`` to the constructor directly when the
            level matters for the nowcast.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> z = np.zeros((200, 2))
            >>> for t in range(1, 200):
            ...     z[t] = [[0.5, 0.2], [0.1, 0.6]] @ z[t - 1] + rng.standard_normal(2)
            >>> fitted = VAR(z, order=1).fit()
            >>> model = MFVAR.from_result(fitted, kinds=["high", "flow"], period=3)
            >>> model.order, model.k_endog
            (1, 2)
            >>> MFVAR.from_result(object(), kinds=["high", "flow"], period=3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: result does not expose ['coefficients', ...
        """
        missing = [a for a in ("coefficients", "sigma_u", "names") if not hasattr(result, a)]
        if missing:
            raise SpecificationError(
                f"result does not expose {missing}; MFVAR.from_result needs a "
                "fitted vector autoregression."
            )
        return cls(
            result.coefficients,  # type: ignore[attr-defined]
            result.sigma_u,  # type: ignore[attr-defined]
            kinds=kinds,
            period=period,
            weights=weights,
            names=tuple(result.names),  # type: ignore[attr-defined]
        )

    @property
    def k_endog(self) -> int:
        """Number of variables.

        Example:
            >>> import numpy as np
            >>> MFVAR(np.eye(3) * 0.5, np.eye(3), kinds=["high", "flow", "stock"], period=3).k_endog
            3
        """
        return int(self._coefficients.shape[1])

    @property
    def order(self) -> int:
        """Latent autoregressive order.

        Example:
            >>> import numpy as np
            >>> A = np.stack([np.eye(2) * 0.4, np.eye(2) * 0.2])
            >>> MFVAR(A, np.eye(2), kinds=["high", "flow"], period=3).order
            2
        """
        return int(self._coefficients.shape[0])

    def _state_space(self) -> _LinearGaussianStateSpace:
        """Assemble the observable form.

        Builds the stacked system through ``_mixed_frequency_system`` -- a
        companion transition over ``max(order, period)`` lags, a selection
        onto the current block, and a constant design matrix carrying each
        variable's aggregation weights -- and wraps it in the package's
        linear Gaussian state space.

        Returns:
            The state-space model whose ``filter`` and ``smooth`` the public
            methods call.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> model = MFVAR(A, S, kinds=["high", "flow"], period=3)
            >>> system = model._state_space()
            >>> type(system).__name__, system.filter(np.zeros((4, 2))).filtered_state.shape
            ('_LinearGaussianStateSpace', (4, 6))
        """
        design, obs_cov, transition, selection, state_cov, state_intercept = (
            _mixed_frequency_system(
                self._coefficients,
                self._sigma_u,
                kinds=self._kinds,
                period=self._period,
                weights=self._weights,
                intercept=self._intercept,
            )
        )
        return _LinearGaussianStateSpace(
            design,
            obs_cov,
            transition,
            selection,
            state_cov,
            state_intercept=state_intercept,
        )

    def filter(self, endog: npt.ArrayLike) -> _KalmanFilterResult:
        """Run the forward pass over the mixed-frequency sample.

        The one-sided estimate: each period's state uses readings up to that
        period only, which is what a real-time nowcast at that date would
        have had.

        Args:
            endog: The ``(nobs, k)`` sample at the high frequency, ``nan``
                wherever a variable is not observed.

        Returns:
            The Kalman filter output over the stacked state, whose first
            ``k`` state columns are the current latent vector.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> model = MFVAR(A, S, kinds=["high", "flow"], period=3)
            >>> filtered = model.filter(y)
            >>> filtered.filtered_state.shape, filtered.filtered_state_cov.shape
            ((24, 6), (24, 6, 6))
            >>> bool(np.isclose(filtered.loglikelihood, model.smooth(y).loglikelihood))
            True
        """
        return self._state_space().filter(_validate_observed(endog))

    def smooth(self, endog: npt.ArrayLike) -> MFVARResult:
        """Infer the latent high-frequency path from the whole sample.

        Args:
            endog: The ``(nobs, k)`` sample at the high frequency, ``nan``
                wherever a variable is not observed.

        Returns:
            The inferred path and its uncertainty, carrying the full
            propagation surface: because the smoothed path is complete, the
            forecast, the impulse responses, and the decompositions read it
            exactly as a VAR result reads its data.

        Raises:
            SpecificationError: If a variable has no observations at all,
                from the sample validation.
            DimensionError: If the sample has no rows beyond the
                autoregressive order, leaving no innovation to compute.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> res = MFVAR(A, S, kinds=["high", "flow"], period=3, intercept=[0.1, 0.0]).smooth(y)
            >>> res.endog.shape, res.resid.shape, res.trend, res.deterministic
            ((24, 2), (23, 2), 'c', array([[0.1, 0. ]]))
            >>> bool(np.allclose(res.endog[:, 0], y[:, 0]))
            True
        """
        observed = _validate_observed(endog)
        model = self._state_space()
        smoothed: _DurbinKoopmanSmootherResult = model.smooth(observed)
        size = self.k_endog
        path = smoothed.smoothed_state[:, :size].copy()
        effective = path.shape[0] - self.order
        if effective < 1:
            raise DimensionError(
                f"a sample of {path.shape[0]} periods leaves no innovations at "
                f"order {self.order}; supply at least {self.order + 1} rows."
            )
        fitted = np.zeros((effective, size), dtype=np.float64)
        if self._intercept is not None:
            fitted += self._intercept
        for i in range(self.order):
            fitted += path[self.order - 1 - i : path.shape[0] - 1 - i] @ self._coefficients[i].T
        stored = (
            None
            if self._weights is None
            else tuple(
                None if w is None else np.asarray(w, dtype=np.float64).ravel()
                for w in self._weights
            )
        )
        return MFVARResult(
            observed=observed,
            endog=path,
            latent_cov=smoothed.smoothed_state_cov[:, :size, :size].copy(),
            coefficients=self._coefficients,
            deterministic=(
                np.zeros((0, size), dtype=np.float64)
                if self._intercept is None
                else self._intercept.reshape(1, size).copy()
            ),
            sigma_u=self._sigma_u,
            resid=path[self.order :] - fitted,
            loglikelihood=model.loglikelihood(observed),
            kinds=self._kinds,
            weights=stored,
            period=self._period,
            names=self._names,
            order=self.order,
            trend="n" if self._intercept is None else "c",
            nobs=effective,
        )

    def nowcast(self, endog: npt.ArrayLike) -> dict[str, float]:
        """The latent value of every variable in the final period.

        Runs the full smoother and reads its last row, so the answer uses
        every reading in the sample; for the one-sided real-time estimate,
        read :meth:`filter` instead.

        Args:
            endog: The mixed-frequency sample.

        Returns:
            Mapping from variable name to its inferred current value.

        Example:
            >>> import numpy as np
            >>> A, S = np.array([[[0.5, 0.1], [0.0, 0.4]]]), np.eye(2)
            >>> y = np.full((24, 2), np.nan)
            >>> rng = np.random.default_rng(0)
            >>> y[:, 0], y[2::3, 1] = rng.standard_normal(24), rng.standard_normal(8)
            >>> model = MFVAR(A, S, kinds=["high", "flow"], period=3)
            >>> model.nowcast(y) == model.smooth(y).nowcast(-1)
            True
        """
        return self.smooth(endog).nowcast(-1)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MIDASVARResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted MIDAS vector autoregression.

    The low-frequency system

    .. math::

       y_t = D d_t + \sum_{i=1}^{p} A_i\, y_{t-i} + B\, x_t(\theta) + u_t,
       \qquad
       x_{t}(\theta)_q = \sum_{j=1}^{L} w_j(\theta_q)\, x^{(q)}_{t, j},

    where :math:`x^{(q)}_{t,j}` is the :math:`j`-th most recent sub-period
    reading of high-frequency series :math:`q` at the end of low-frequency
    period :math:`t`, and the weights are the normalized exponential Almon
    curve :math:`w_j \propto \exp(\theta_1 j + \theta_2 j^2)` summing to one,
    so the scale of each regressor lives in its column of :math:`B` and the
    two entries of :math:`\theta_q` carry only the shape of the window.
    Estimation profiles :math:`\theta` -- least squares conditional on it,
    a derivative-free search over it -- and this record holds the estimate
    at the maximizer.

    The reduced-form surface is inherited unchanged, and everything it reports
    is conditional in exactly one place: the coefficient standard errors, and
    every test built from them, take the estimated lag polynomial as known.
    :meth:`joint_stderr` is the honest alternative and says what it costs.

    :meth:`forecast` is conditional in the same sense a VARX forecast is: the
    model holds no process for the high-frequency block, so it asks for the
    future sub-period path and refuses without it.

    Note:
        The propagation surface -- companion, stability, impulse responses,
        variance and historical decompositions -- reads the endogenous block
        only. The compressed regressors are conditioned on, not shocked, so
        they enter no moving-average representation, and a response here is
        the response of the low-frequency system to its own innovations
        holding the high-frequency path fixed. :attr:`exog` is the compressed
        block at the *estimated* polynomial, so it is a function of the data
        and of :attr:`theta`, not a supplied regressor; the inherited
        ``design`` treats it as data, which is the conditioning the Note
        above describes.

    Attributes:
        endog: The low-frequency sample.
        exog_high: The high-frequency panel as supplied, chronological.
        exog: The compressed regressor block over the effective sample, one
            column per high-frequency series at the estimated polynomial.
        names: Endogenous labels, in Cholesky order.
        exog_names: High-frequency series labels.
        order: Endogenous autoregressive order.
        period: Sub-periods per low-frequency period.
        midas_lags: Sub-periods each weight curve spans.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        midas_coefficients: ``(k, m)`` slopes on the compressed regressors.
        theta: ``(m, 2)`` estimated lag-polynomial parameters, one row per
            high-frequency series.
        deterministic: Deterministic coefficients, one row per term.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: Residuals over the effective sample.
        fittedvalues: One-step conditional means over the effective sample.
        design: The regressor matrix as estimated, compressed block last.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Free parameters -- coefficients, covariance, and the two
            polynomial parameters per high-frequency series.

    See Also:
        * :class:`MIDASVAR` -- the model whose ``fit()`` returns this record.
        * :class:`MFVARResult` -- the high-frequency alternative, inferred
          rather than estimated.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARXResult`
          -- the same conditional surface with an unrestricted exogenous
          block, the model this collapses to at ``period == 1``.

    References:
        Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch:
        Mixed data sampling regression models. CIRANO Working Paper 2004s-20.

        Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS regressions:
        Further results and new directions. *Econometric Reviews*, 26(1),
        53-90.

        Ghysels, E. (2016). Macroeconomics and the reality of mixed frequency
        data. *Journal of Econometrics*, 193(2), 294-314.

    Example:
        A quarterly pair in which the first variable loads on the last two
        months of a monthly series with weights 0.7 and 0.3. The profiled
        polynomial recovers the decaying curve and a unit slope, the second
        equation's slope is nothing, the joint standard errors widen the
        conditional ones where the polynomial matters, and the forecast
        needs the three future months:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((360, 1))
        >>> window = 0.7 * x[2::3, 0] + 0.3 * x[1::3, 0]
        >>> noise = 0.1 * rng.standard_normal(120)
        >>> y = np.column_stack([window + noise, rng.standard_normal(120)])
        >>> res = MIDASVAR(y, x, order=1, period=3).fit()
        >>> res.midas_weights.ravel().round(2), res.midas_coefficients.round(1).ravel() + 0.0
        (array([0.69, 0.3 , 0.01]), array([1., 0.]))
        >>> res.nobs, res.n_params, res.k_exog, res.midas_lags
        (119, 13.0, 1, 3)
        >>> joint = res.joint_stderr()
        >>> bool(joint["y1: x1"] > res.bse["y1: x1"]), sorted(k for k in joint if "theta" in k)
        (True, ['theta1[x1]', 'theta2[x1]'])
        >>> res.forecast(1, exog_high_future=np.ones((3, 1))).shape
        (1, 2)
    """

    coefficients: npt.NDArray[np.float64]
    """The ``(p, k, k)`` endogenous autoregressive matrices."""

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` deterministic coefficients, one row per term."""

    exog_high: npt.NDArray[np.float64]
    """The ``(nobs_low * period, m)`` high-frequency panel as supplied, chronological."""

    exog: npt.NDArray[np.float64]
    """The ``(nobs, m)`` compressed regressors at the estimated polynomial."""

    exog_names: tuple[str, ...]
    """High-frequency series labels."""

    midas_coefficients: npt.NDArray[np.float64]
    r"""The ``(k, m)`` slopes :math:`B` on the compressed regressors."""

    theta: npt.NDArray[np.float64]
    r"""The ``(m, 2)`` polynomial parameters :math:`(\theta_1, \theta_2)` per series."""

    period: int
    """Sub-periods per low-frequency period."""

    midas_lags: int
    """Sub-periods each weight curve spans."""

    @property
    def k_exog(self) -> int:
        """Number of high-frequency series.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 2))
            >>> y = np.column_stack([x[2::3, 0] + 0.1 * rng.standard_normal(80), x[1::3, 1]])
            >>> MIDASVAR(y, x, order=1, period=3).fit().k_exog
            2
        """
        return len(self.exog_names)

    @property
    def midas_weights(self) -> npt.NDArray[np.float64]:
        """The estimated weight curves, ``(midas_lags, m)``, most recent first.

        Each column sums to one; a column that decays toward zero within the
        window is the sign that ``midas_lags`` is long enough.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((360, 1))
            >>> window = 0.7 * x[2::3, 0] + 0.3 * x[1::3, 0]
            >>> noise = 0.1 * rng.standard_normal(120)
            >>> y = np.column_stack([window + noise, rng.standard_normal(120)])
            >>> weights = MIDASVAR(y, x, order=1, period=3, midas_lags=6).fit().midas_weights
            >>> weights.shape, bool(np.isclose(weights.sum(), 1.0)), weights.ravel().round(2)
            ((6, 1), True, array([0.69, 0.3 , 0.01, 0.  , 0.  , 0.  ]))
        """
        return np.column_stack([_midas_weights(row, self.midas_lags) for row in self.theta])

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The compressed-regressor coefficients, in design order.

        Returns:
            ``(midas_coefficients.T,)``, an ``(m, k)`` block whose rows match
            the compressed columns at the end of :attr:`design`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 1))
            >>> last = x[2::3, 0] + 0.1 * rng.standard_normal(80)
            >>> y = np.column_stack([last, rng.standard_normal(80)])
            >>> res = MIDASVAR(y, x, order=1, period=3).fit()
            >>> [block.shape for block in res._trailing_blocks()]
            [(1, 2)]
        """
        return (self.midas_coefficients.T,)

    def _trailing_labels(self) -> tuple[str, ...]:
        """Compressed-regressor column names.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 1))
            >>> last = x[2::3, 0] + 0.1 * rng.standard_normal(80)
            >>> y = np.column_stack([last, rng.standard_normal(80)])
            >>> res = MIDASVAR(y, x, order=1, period=3, exog_names=["ip"]).fit()
            >>> res._trailing_labels(), res._regressor_labels()
            (('ip',), ('const', 'y1.L1', 'y2.L1', 'ip'))
        """
        return self.exog_names

    def forecast(
        self, steps: int = 1, *, exog_high_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Point forecasts conditional on a future high-frequency path.

        The supplied path is appended to :attr:`exog_high`, compressed through
        the estimated weight curves one low-frequency period at a time, and
        the endogenous block is iterated from the last ``order`` observations.

        Args:
            steps: Horizon in low-frequency periods.
            exog_high_future: A ``(steps * period, m)`` chronological path for
                the high-frequency block, in the column order of
                :attr:`exog_names`. Required.

        Returns:
            A ``(steps, k)`` array of conditional means.

        Raises:
            SpecificationError: If ``steps`` is not positive, or
                ``exog_high_future`` is omitted.
            DimensionError: If ``exog_high_future`` has the wrong shape.

        Example:
            A unit path for the next quarter's three months moves the first
            variable by roughly its unit slope:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((360, 1))
            >>> window = 0.7 * x[2::3, 0] + 0.3 * x[1::3, 0]
            >>> noise = 0.1 * rng.standard_normal(120)
            >>> y = np.column_stack([window + noise, rng.standard_normal(120)])
            >>> res = MIDASVAR(y, x, order=1, period=3).fit()
            >>> path = res.forecast(2, exog_high_future=np.ones((6, 1)))
            >>> path.shape, bool(abs(path[0, 0] - 1.0) < 0.15)
            ((2, 2), True)
            >>> res.forecast(2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a MIDASVAR forecast is conditional on the ...
            >>> res.forecast(2, exog_high_future=np.ones((6, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: exog_high_future has 2 columns but the model ...
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        if exog_high_future is None:
            raise SpecificationError(
                "a MIDASVAR forecast is conditional on the high-frequency path, so it "
                "cannot be produced from the fitted model alone: pass exog_high_future "
                f"with {steps * self.period} rows ({self.period} sub-periods per step) "
                f"and {self.k_exog} columns for {self.exog_names}. This model holds no "
                "process for the high-frequency block and will not invent one, because "
                "doing so would turn a conditional forecast into an unconditional "
                "forecast without saying so."
            )
        future = validate_exog_matrix(
            exog_high_future, nobs=steps * self.period, label="exog_high_future"
        )
        if future.shape[1] != self.k_exog:
            raise DimensionError(
                f"exog_high_future has {future.shape[1]} columns but the model was "
                f"fitted with {self.k_exog}."
            )
        k, p = self.k_endog, self.order
        low = self.endog.shape[0]
        full = np.vstack([self.exog_high, future])
        windows = _midas_windows(
            full, nobs=low + steps, period=self.period, lags=self.midas_lags, start=low
        )
        compressed = np.einsum("tjm,jm->tm", windows, self.midas_weights)
        det = deterministic_columns(self.trend, steps, start=low + 1)
        history = [self.endog[low - i - 1] for i in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = det[h] @ self.deterministic if self.deterministic.shape[0] else np.zeros(k)
            for i in range(p):
                point = point + self.coefficients[i] @ history[i]
            point = point + self.midas_coefficients @ compressed[h]
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    def joint_stderr(self) -> dict[str, float]:
        r"""Standard errors that also account for estimating the lag polynomial.

        The default surface -- ``bse``, ``tvalues``, ``conf_int``, and the
        coefficient table -- conditions on the estimated polynomial, treating
        the compressed regressor as data. Here the observed information of the
        covariance-concentrated Gaussian likelihood,

        .. math::

           \ell_c(\beta, \theta) = -\tfrac{T}{2}
           \log\det\bigl(\hat\Sigma(\beta, \theta)\bigr) + \text{const},

        is computed numerically over the full parameter vector, every
        regression coefficient and every polynomial parameter jointly, and
        inverted. The profile-likelihood curvature is the correct marginal
        information for the retained parameters, so concentrating the
        covariance out changes nothing about the answer while removing
        ``k(k+1)/2`` dimensions from the Hessian.

        The cost is quadratic in the parameter count -- each Hessian entry is
        four likelihood evaluations -- which is why this is a method rather
        than the default.

        Returns:
            Mapping from parameter to its joint standard error: every key of
            :attr:`params`, plus ``"theta1[x]"`` and ``"theta2[x]"`` per
            high-frequency series.

        Raises:
            NumericalError: If the observed information is not positive
                definite at the estimate, which usually means the polynomial is
                weakly identified -- a flat weight curve makes ``theta``
                unidentifiable and its rows of the information singular.

        Example:
            The slope on the compressed regressor is where the polynomial
            matters, so its joint error exceeds the conditional one; the
            autoregressive coefficients barely move:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((360, 1))
            >>> window = 0.7 * x[2::3, 0] + 0.3 * x[1::3, 0]
            >>> noise = 0.1 * rng.standard_normal(120)
            >>> y = np.column_stack([window + noise, rng.standard_normal(120)])
            >>> res = MIDASVAR(y, x, order=1, period=3).fit()
            >>> joint = res.joint_stderr()
            >>> len(joint), len(res.params) + 2
            (10, 10)
            >>> bool(joint["y1: x1"] > res.bse["y1: x1"])
            True
            >>> bool(abs(joint["y1: y1.L1"] / res.bse["y1: y1.L1"] - 1.0) < 0.1)
            True
        """
        k, m = self.k_endog, self.k_exog
        width = self.design.shape[1]
        effective = int(self.nobs)
        burn = self.endog.shape[0] - effective
        target = self.endog[burn:]
        base = self.design[:, : width - m]
        windows = _midas_windows(
            self.exog_high,
            nobs=self.endog.shape[0],
            period=self.period,
            lags=self.midas_lags,
            start=burn,
        )
        blocks = [self.deterministic]
        blocks.extend(self.coefficients[i].T for i in range(self.order))
        blocks.append(self.midas_coefficients.T)
        point = np.concatenate([np.vstack(blocks).ravel(), self.theta.ravel()])

        def criterion(vector: npt.NDArray[np.float64]) -> float:
            beta = vector[: width * k].reshape(width, k)
            packed = vector[width * k :].reshape(m, 2)
            curves = np.column_stack([_midas_weights(row, self.midas_lags) for row in packed])
            compressed = np.einsum("tjm,jm->tm", windows, curves)
            resid = target - np.column_stack([base, compressed]) @ beta
            sign, logdet = np.linalg.slogdet(resid.T @ resid / effective)
            if sign <= 0:
                raise NumericalError(
                    "the residual covariance left the positive definite cone while "
                    "differentiating; the joint information is not defined here."
                )
            return 0.5 * effective * float(logdet)

        count = point.shape[0]
        steps = 1e-4 * np.maximum(1.0, np.abs(point))
        hessian = np.empty((count, count), dtype=np.float64)
        for i in range(count):
            for j in range(i, count):
                pp = point.copy()
                pp[i] += steps[i]
                pp[j] += steps[j]
                pm = point.copy()
                pm[i] += steps[i]
                pm[j] -= steps[j]
                mp = point.copy()
                mp[i] -= steps[i]
                mp[j] += steps[j]
                mm = point.copy()
                mm[i] -= steps[i]
                mm[j] -= steps[j]
                value = (criterion(pp) - criterion(pm) - criterion(mp) + criterion(mm)) / (
                    4.0 * steps[i] * steps[j]
                )
                hessian[i, j] = value
                hessian[j, i] = value
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the observed information is singular at the estimate; the joint "
                "standard errors are not identified."
            ) from error
        variances = np.diagonal(covariance)
        if np.any(variances <= 0):
            raise NumericalError(
                "the observed information is not positive definite at the estimate; "
                "the joint standard errors are not identified. This usually means "
                "the lag polynomial is weakly identified -- a flat weight curve "
                "makes theta unidentifiable."
            )
        stderr = np.sqrt(variances)
        out: dict[str, float] = {}
        for r, regressor in enumerate(self._regressor_labels()):
            for e, equation in enumerate(self.names):
                out[f"{equation}: {regressor}"] = float(stderr[r * k + e])
        offset = width * k
        for q, source in enumerate(self.exog_names):
            out[f"theta1[{source}]"] = float(stderr[offset + 2 * q])
            out[f"theta2[{source}]"] = float(stderr[offset + 2 * q + 1])
        return out

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Returns:
            ``"MIDASVAR(p, L)"``, with ``, trend=<trend>`` appended when the
            trend is not the default constant.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 1))
            >>> last = x[2::3, 0] + 0.1 * rng.standard_normal(80)
            >>> y = np.column_stack([last, rng.standard_normal(80)])
            >>> MIDASVAR(y, x, order=1, period=3).fit()._comparison_label()
            'MIDASVAR(1, 3)'
            >>> MIDASVAR(y, x, order=1, period=3, trend="n").fit()._comparison_label()
            'MIDASVAR(1, 3, trend=n)'
        """
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"MIDASVAR({self.order}, {self.midas_lags}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        The coefficient table with its conditional inference columns, the
        compressed block labelled by series; metadata pairing the fit
        statistics with the dimensions; notes on stability, the estimated
        polynomial per series, the endogenous-only propagation surface, the
        conditional forecast, the conditioning of the standard errors, and
        the Cholesky ordering, with an unstable system flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 1))
            >>> last = x[2::3, 0] + 0.1 * rng.standard_normal(80)
            >>> y = np.column_stack([last, rng.standard_normal(80)])
            >>> table = MIDASVAR(y, x, order=1, period=3).fit()._summary_table()
            >>> table.title, dict(table.metadata)["High-frequency"]
            ('MIDASVAR(1, 3) Results', '1')
            >>> [row[0] for row in table.rows][:4]
            ['y1: const', 'y1: y1.L1', 'y1: y2.L1', 'y1: x1']
            >>> table.notes[1][:50]
            'Lag polynomial (theta_1, theta_2) over 3 sub-perio'
        """
        criteria = self.information_criteria
        stability = self.stability_check()
        polynomial = ", ".join(
            f"{name}: ({row[0]:+.3f}, {row[1]:+.3f})"
            for name, row in zip(self.exog_names, self.theta, strict=True)
        )
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            f"Lag polynomial (theta_1, theta_2) over {self.midas_lags} sub-periods: {polynomial}.",
            "Stability, impulse responses, and the variance decomposition read the "
            "low-frequency endogenous block only. The high-frequency regressors are "
            "conditioned on rather than shocked, so they do not enter the "
            "moving-average representation.",
            "Forecasts are conditional: forecast() requires the future high-frequency "
            "path and will not extrapolate it.",
            _MIDAS_CONDITIONAL_NOTE,
            _CHOLESKY_NOTE,
        ]
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        return SummaryTable(
            title=f"MIDASVAR({self.order}, {self.midas_lags}) Results",
            metadata=(
                ("Model", f"MIDASVAR({self.order}, {self.midas_lags})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Endogenous", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("High-frequency", f"{self.k_exog}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Sub-periods", f"{self.period}"),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Trend", self.trend),
                ("Observations", f"{self.nobs}"),
            ),
            columns=self._coefficient_columns(),
            rows=self._coefficient_rows(),
            notes=tuple(notes),
        )


class MIDASVAR(_VectorAutoRegressionModel[MIDASVARResult]):
    r"""A low-frequency VAR reading high-frequency data through a lag polynomial.

    The model is :math:`y_t = D d_t + A_1 y_{t-1} + \dots + A_p y_{t-p} + B\,
    x_t(\theta) + u_t` at the low frequency, where :math:`x_t(\theta)`
    compresses the last ``midas_lags`` sub-period readings of each
    high-frequency series into one number with normalized exponential-Almon
    weights :math:`w_j \propto \exp(\theta_1 j + \theta_2 j^2)`. Two
    parameters per series buy the whole within-period history, which is the
    entire point of MIDAS: the unrestricted alternative spends ``midas_lags``
    coefficients per series per equation and eats the sample.

    **This class estimates**, unlike its state-space sibling, because nothing
    here is temporally aggregated: every regressor is observed, the model is
    linear conditional on the polynomial, and the likelihood identifies
    everything it touches. Estimation profiles the polynomial -- an inner
    least-squares solve conditional on ``theta``, an outer derivative-free
    search over it, multi-started because the decay direction can be
    multi-modal. Lag-order selection holds the weights flat so that the
    candidates differ in the endogenous order and in nothing else.

    Attributes:
        _endog: The validated ``(nobs, k)`` low-frequency panel.
        _names: Endogenous labels.
        _order: Endogenous autoregressive order.
        _trend: Deterministic terms.
        _prior: The base slot for a shrinkage prior, unused by this family.
        _exog_high: The validated ``(nobs * period, m)`` high-frequency panel.
        _exog_names: High-frequency labels.
        _period: Sub-periods per low-frequency period.
        _midas_lags: Sub-periods each weight curve spans.

    See Also:
        * :class:`MIDASVARResult` -- the record ``fit()`` returns.
        * :class:`MFVAR` -- the high-frequency model, which infers a latent
          path from supplied parameters instead of estimating.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
          -- the right model when ``period`` would be one.

    References:
        Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch:
        Mixed data sampling regression models. CIRANO Working Paper 2004s-20.

        Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS regressions:
        Further results and new directions. *Econometric Reviews*, 26(1),
        53-90.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((120, 1))
        >>> y = np.column_stack(
        ...     [x[2::3, 0] + 0.1 * rng.standard_normal(40), rng.standard_normal(40)]
        ... )
        >>> res = MIDASVAR(y, x, order=1, period=3).fit()
        >>> res.forecast(2, exog_high_future=np.zeros((6, 1))).shape
        (2, 2)
    """

    __slots__ = ("_exog_high", "_exog_names", "_midas_lags", "_period")

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog_high: npt.ArrayLike,
        *,
        order: int,
        period: int,
        midas_lags: int | None = None,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
    ) -> None:
        """Validate both panels and the specification.

        Args:
            endog: The ``(nobs, k)`` low-frequency panel.
            exog_high: The ``(nobs * period, m)`` high-frequency panel,
                chronological, exactly ``period`` sub-period rows per
                low-frequency observation. Missing values are rejected: a
                ragged calendar is :class:`MFVAR`'s problem statement, not
                this model's.
            order: Endogenous autoregressive order.
            period: Sub-periods per low-frequency period, at least two -- with
                one sub-period nothing is mixed and
                :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
                is the right model.
            midas_lags: Sub-periods each weight curve spans. Defaults to
                ``period``; longer windows reach into earlier low-frequency
                periods and cost leading observations.
            trend: Deterministic terms.
            names: Endogenous labels. Defaults to ``y1 ... yk``.
            exog_names: High-frequency labels. Defaults to ``x1 ... xm``.

        Raises:
            SpecificationError: If ``period`` or ``midas_lags`` is malformed,
                or the labels are malformed or collide.
            DimensionError: If the two panels do not align at ``period`` rows
                per observation, or the sample is too short for the
                specification.
            NumericalError: If either panel is non-finite.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((240, 1))
            >>> y = rng.standard_normal((80, 2))
            >>> model = MIDASVAR(y, x, order=1, period=3, midas_lags=6, exog_names=["ip"])
            >>> model.period, model.midas_lags, model.exog_names, model.k_exog
            (3, 6, ('ip',), 1)
            >>> MIDASVAR(y, x, order=1, period=1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: period must be an integer >= 2; got 1. ...
            >>> MIDASVAR(y, x, order=1, period=4)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: exog_high has 240 rows but the endogenous panel ...
            >>> MIDASVAR(y, x, order=1, period=3, names=["x1", "b"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: names and exog_names must not overlap, ...
        """
        if int(period) != period or period < 2:
            raise SpecificationError(
                f"period must be an integer >= 2; got {period!r}. With one sub-period "
                "nothing is mixed, and VARX is that model."
            )
        self._period = int(period)
        rows = int(np.asarray(endog, dtype=np.float64).shape[0])
        self._exog_high = validate_exog_matrix(
            exog_high, nobs=rows * self._period, label="exog_high"
        )
        lags = self._period if midas_lags is None else midas_lags
        if int(lags) != lags or lags < 1:
            raise SpecificationError(f"midas_lags must be an integer >= 1; got {lags!r}.")
        self._midas_lags = int(lags)
        self._exog_names = self._resolve_names(
            exog_names, self._exog_high.shape[1], "exog_names", "x"
        )
        super().__init__(endog, order=order, trend=trend, names=names)
        overlap = set(self._names) & set(self._exog_names)
        if overlap:
            raise SpecificationError(
                "names and exog_names must not overlap, or a coefficient table cannot "
                f"say which block a row came from; both contain {tuple(sorted(overlap))}."
            )

    @property
    def exog_high(self) -> npt.NDArray[np.float64]:
        """The validated high-frequency panel.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 1)), rng.standard_normal((20, 2))
            >>> MIDASVAR(y, x, order=1, period=3).exog_high.shape
            (60, 1)
        """
        return self._exog_high

    @property
    def exog_names(self) -> tuple[str, ...]:
        """High-frequency labels.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 2)), rng.standard_normal((20, 2))
            >>> MIDASVAR(y, x, order=1, period=3).exog_names
            ('x1', 'x2')
        """
        return self._exog_names

    @property
    def k_exog(self) -> int:
        """Number of high-frequency series.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 2)), rng.standard_normal((20, 2))
            >>> MIDASVAR(y, x, order=1, period=3).k_exog
            2
        """
        return int(self._exog_high.shape[1])

    @property
    def period(self) -> int:
        """Sub-periods per low-frequency period.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 1)), rng.standard_normal((20, 2))
            >>> MIDASVAR(y, x, order=1, period=3).period
            3
        """
        return self._period

    @property
    def midas_lags(self) -> int:
        """Sub-periods each weight curve spans.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 1)), rng.standard_normal((20, 2))
            >>> MIDASVAR(y, x, order=1, period=3).midas_lags
            3
            >>> MIDASVAR(y, x, order=1, period=3, midas_lags=5).midas_lags
            5
        """
        return self._midas_lags

    @property
    def n_regressors(self) -> int:
        """Regressors per equation, including the compressed block.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 2)), rng.standard_normal((20, 3))
            >>> MIDASVAR(y, x, order=1, period=3).n_regressors
            6
        """
        return super().n_regressors + self.k_exog

    def _burn_for(self, order: int) -> int:
        """Leading observations lost to the lags or to the weight window.

        A window of ``midas_lags`` sub-periods ending in low-frequency period
        ``t`` reaches back ``ceil(midas_lags / period) - 1`` periods, and the
        burn is the larger of that reach and the endogenous order.

        Args:
            order: Endogenous order to compute the burn for.

        Returns:
            The number of leading low-frequency rows the design cannot use.

        Example:
            >>> import numpy as np
            >>> x = np.arange(24.0).reshape(24, 1)
            >>> model = MIDASVAR(np.zeros((8, 2)), x, order=1, period=3, midas_lags=7)
            >>> model._burn_for(1), model._burn_for(3)
            (2, 3)
        """
        reach = (self._midas_lags + self._period - 1) // self._period - 1
        return max(order, reach)

    def _max_supported_lags(self) -> int:
        """Largest endogenous order the sample can identify.

        Returns:
            The order at which the design would still have more rows than
            columns, after the deterministic terms and the compressed block
            are counted.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((120, 1))
            >>> MIDASVAR(rng.standard_normal((40, 2)), x, order=1, period=3)._max_supported_lags()
            12
        """
        free = int(self._endog.shape[0]) - self._n_deterministic_columns - self.k_exog - 1
        return max(free // (self.k_endog + 1), 0)

    def _design(
        self, order: int | None = None, *, trim: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Build the target and regressor matrix at flat weights.

        Used by lag-order selection, where the candidates must differ in the
        endogenous order and in nothing else: holding the polynomial flat keeps
        the compressed block identical across candidates, so the criteria
        compare lag structures rather than incidental re-estimates of the
        weights.

        Args:
            order: Endogenous order; defaults to the fitted order.
            trim: Leading low-frequency observations to discard first.

        Returns:
            The target block, the design, and the row count of both.

        Raises:
            DimensionError: If the trimmed sample cannot supply the lags and
                the weight window.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((120, 1))
            >>> model = MIDASVAR(rng.standard_normal((40, 2)), x, order=1, period=3)
            >>> target, design, effective = model._design(order=2)
            >>> target.shape, design.shape, effective
            ((38, 2), (38, 6), 38)
            >>> bool(np.allclose(design[:, -1], x[:, 0].reshape(40, 3)[2:].mean(axis=1)))
            True
        """
        lags = self._order if order is None else order
        burn = self._burn_for(lags)
        panel = self._endog[trim:]
        high = self._exog_high[trim * self._period :]
        nobs = panel.shape[0]
        if nobs <= burn:
            raise DimensionError(
                f"{nobs} observations is too few for a design of order {lags} with a "
                f"{self._midas_lags}-sub-period weight window."
            )
        effective = nobs - burn
        det = deterministic_columns(self._trend, effective, start=trim + burn + 1)
        windows = _midas_windows(
            high, nobs=nobs, period=self._period, lags=self._midas_lags, start=burn
        )
        flat = np.full(self._midas_lags, 1.0 / self._midas_lags, dtype=np.float64)
        compressed = np.einsum("tjm,j->tm", windows, flat)
        design = np.column_stack([det, lag_matrix(panel, lags, start=burn), compressed])
        return panel[burn:], design, effective

    def _blocks(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Assemble the fixed pieces the profile search reuses at every draw.

        Returns:
            The target block, the deterministic-and-lag design, and the
            ``(effective, midas_lags, m)`` window stack.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((120, 1))
            >>> model = MIDASVAR(rng.standard_normal((40, 2)), x, order=1, period=3)
            >>> target, base, windows = model._blocks()
            >>> target.shape, base.shape, windows.shape
            ((39, 2), (39, 3), (39, 3, 1))
            >>> bool(np.allclose(windows[0, :, 0], x[5:2:-1, 0]))
            True
        """
        burn = self._burn_for(self._order)
        nobs = self._endog.shape[0]
        effective = nobs - burn
        det = deterministic_columns(self._trend, effective, start=burn + 1)
        base = np.column_stack([det, lag_matrix(self._endog, self._order, start=burn)])
        windows = _midas_windows(
            self._exog_high,
            nobs=nobs,
            period=self._period,
            lags=self._midas_lags,
            start=burn,
        )
        return self._endog[burn:], base, windows

    def _seeds(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Starting points: flat weights, gentle decay, and steep decay.

        Returns:
            Three ``(2 * m,)`` vectors with ``theta_1 = 0`` for every series
            and ``theta_2`` in ``(0.0, -0.05, -0.5)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x, y = rng.standard_normal((60, 2)), rng.standard_normal((20, 2))
            >>> [seed.tolist() for seed in MIDASVAR(y, x, order=1, period=3)._seeds()]
            [[0.0, 0.0, 0.0, 0.0], [0.0, -0.05, 0.0, -0.05], [0.0, -0.5, 0.0, -0.5]]
        """
        return tuple(
            np.tile(np.array([0.0, decay], dtype=np.float64), self.k_exog)
            for decay in (0.0, -0.05, -0.5)
        )

    def fit(self) -> MIDASVARResult:
        """Estimate the system by profiled maximum likelihood.

        Conditional on the polynomial parameters the model is linear, so the
        inner problem is one least-squares solve with the covariance
        concentrated out; the outer search is derivative-free over the two
        parameters per high-frequency series, multi-started because the decay
        direction can be multi-modal.

        Returns:
            The fitted result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal((360, 1))
            >>> window = 0.7 * x[2::3, 0] + 0.3 * x[1::3, 0]
            >>> noise = 0.1 * rng.standard_normal(120)
            >>> y = np.column_stack([window + noise, rng.standard_normal(120)])
            >>> res = MIDASVAR(y, x, order=1, period=3).fit()
            >>> res.theta.shape, res.exog.shape, res.design.shape
            ((1, 2), (119, 1), (119, 4))
            >>> bool(np.allclose(res.design[:, -1:], res.exog))
            True
        """
        target, base, windows = self._blocks()
        objective = _MidasProfileObjective(
            target=target, base_design=base, windows=windows, seeds=self._seeds()
        )
        theta, _ = _maximize_likelihood(objective)
        design = objective.design(theta.ravel())
        moments = self._gaussian_moments(target, design)
        k, m = self.k_endog, self.k_exog
        head = self._n_deterministic_columns + k * self._order
        return MIDASVARResult(
            endog=self._endog,
            exog_high=self._exog_high,
            exog=design[:, head:].copy(),
            names=self._names,
            exog_names=self._exog_names,
            order=self._order,
            period=self._period,
            midas_lags=self._midas_lags,
            trend=self._trend,
            coefficients=self._lag_blocks(moments.coef),
            midas_coefficients=moments.coef[head : head + m, :].T.copy(),
            theta=theta,
            deterministic=moments.coef[: self._n_deterministic_columns],
            sigma_u=moments.sigma_u,
            sigma_ml=moments.sigma_ml,
            resid=moments.resid,
            fittedvalues=moments.fittedvalues,
            design=design,
            llf=moments.llf,
            nobs=moments.nobs,
            n_params=k * moments.width + k * (k + 1) / 2 + 2 * m,
        )
