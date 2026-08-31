# filepath: /src/cultivars/var/reduced_form/mixed_frequency.py
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

"""Mixed-frequency vector autoregressions: two models for two different problems.

Both models here relate series sampled at different frequencies, and that is
the whole of their resemblance. They differ in which frequency the *model*
lives at, and the difference decides everything else -- estimator, result
surface, and whether a maximum-likelihood ``fit`` is defensible at all.

:class:`MFVAR` puts the model at the **high** frequency. The monthly path is
latent, a quarterly reading is a linear functional of it, and the Kalman filter
recovers the path. This is the Mariano-Murasawa state-space formulation. Its
one structural elegance is that the observation matrix is *constant*: the
calendar is not in ``Z``, it is in the missing-value pattern of the sample, and
the filter already masks missing rows. Nothing about the model changes between
a month in which the quarterly series is observed and one in which it is not.

:class:`MIDASVAR` puts the model at the **low** frequency. The high-frequency
regressor is compressed into one column per low-frequency date by a lag
polynomial with two parameters, so a year of daily history costs two degrees of
freedom rather than two hundred and fifty. This is Ghysels-Santa-Clara-Valkanov.

The asymmetry in what they expose is deliberate and is documented at each
class. :class:`MIDASVAR` estimates. :class:`MFVAR` does not, and the reason is
recorded in its docstring rather than buried in a release note: with a
temporally aggregated variable, the Gaussian likelihood is not merely flat in
that variable's own innovation variance, it is *maximized away from the truth*,
and the error grows with the sample. A model that returns a confident wrong
number is worse than one that declines, so :class:`MFVAR` takes its parameters
and filters with them.

References:
    Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business
        cycles based on monthly and quarterly series. *Journal of Applied
        Econometrics*, 18(4), 427-443.
    Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch:
        Mixed data sampling regression models. Working paper.
    Ghysels, E. (2016). Macroeconomics and the reality of mixed frequency
        data. *Journal of Econometrics*, 193(2), 294-314.
    Schorfheide, F., & Song, D. (2015). Real-time forecasting with a
        mixed-frequency VAR. *Journal of Business & Economic Statistics*,
        33(3), 366-380.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from ..._core import (
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
from ..._internals import (
    _ComparisonMixin,
    _DurbinKoopmanSmootherResult,
    _KalmanFilterResult,
    _LinearGaussianStateSpaceModel,
    _maximize_likelihood,
    _MidasProfileObjective,
    _SummaryMixin,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import DimensionError, NumericalError, SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MFVARResult(_SummaryMixin, _VectorPropagationMixin):
    """The latent high-frequency path implied by a mixed-frequency sample.

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
    """

    observed: npt.NDArray[np.float64] = field(repr=False)
    endog: npt.NDArray[np.float64] = field(repr=False)
    latent_cov: npt.NDArray[np.float64] = field(repr=False)
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    loglikelihood: float
    kinds: tuple[Frequency, ...]
    period: int
    names: tuple[str, ...]
    order: int
    trend: str
    nobs: int
    weights: tuple[npt.NDArray[np.float64] | None, ...] | None = field(default=None, repr=False)

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return len(self.names)

    @property
    def latent_stderr(self) -> npt.NDArray[np.float64]:
        """Pointwise standard errors of the latent path, ``(nobs, k)``.

        The diagonal of a smoothed covariance reaches zero from below by a few
        units in the last place wherever a variable is directly observed, and
        an unguarded square root would return ``nan`` for exactly the entries
        the model knows best. The clip is not cosmetic.
        """
        var = np.diagonal(self.latent_cov, axis1=1, axis2=2)
        return np.sqrt(np.clip(var, 0.0, None))

    @property
    def observed_fraction(self) -> npt.NDArray[np.float64]:
        """Share of periods in which each variable was directly observed."""
        return np.isfinite(self.observed).mean(axis=0)

    def implied_low_frequency(self) -> npt.NDArray[np.float64]:
        """Aggregate the inferred latent path back to the observed frequency.

        Returns:
            An ``(nobs, k)`` array whose entries are the model's fitted value
            for a low-frequency reading ending at that period. Comparing this
            with :attr:`observed` at the periods where a reading exists is the
            residual check for a mixed-frequency model.
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
        """
        nobs = self.endog.shape[0]
        pos = index + nobs if index < 0 else index
        if not 0 <= pos < nobs:
            raise DimensionError(f"index {index} is out of range for {nobs} periods.")
        return {name: float(self.endog[pos, i]) for i, name in enumerate(self.names)}

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
    """A latent high-frequency VAR observed through temporal aggregation.

    The model is ``z_t = c + A_1 z_{t-1} + ... + A_p z_{t-p} + u_t`` at the high
    frequency; what is observed is ``z`` itself for some variables and a
    weighted sub-period sum for others, on a calendar expressed as ``nan`` in
    the sample rather than as a time-varying observation matrix.

    **This class does not estimate.** The omission is a finding, not an
    unfinished feature. Simulating a three-variable monthly VAR with one
    variable observed only as a quarterly average and estimating by EM:

    ==========  ==========  ============  ===========
    ``T``       ``|A err|``  ``Sigma11``   ``|S err|``
    ==========  ==========  ============  ===========
    control      0.051       1.100          0.060
    1500         0.129       2.629          1.469
    6000         0.183       3.104          1.944
    ==========  ==========  ============  ===========

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
    parameters and filter.

    Args:
        coefficients: ``(order, k, k)`` latent autoregressive coefficients, or
            ``(k, k)`` for an order of one.
        sigma_u: ``(k, k)`` latent innovation covariance.
        kinds: One :data:`Frequency` per variable.
        period: High-frequency sub-periods per low-frequency period.
        intercept: Optional ``(k,)`` latent intercept.
        weights: Optional per-variable sub-period weights, most recent first.
        names: Variable labels. Defaults to ``y1 ... yk``.

    Raises:
        DimensionError: If the coefficient or covariance shapes disagree.
        SpecificationError: If ``period`` is not a positive integer, ``kinds``
            has the wrong length or an unrecognized entry, or every variable is
            marked ``high`` -- in which case there is no mixed frequency and a
            plain :class:`~cultivars.var.reduced_form.VAR` is the right model.

    Example:
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
        """Initialize the mixed-frequency VAR."""
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
        """Number of variables."""
        return int(self._coefficients.shape[1])

    @property
    def order(self) -> int:
        """Latent autoregressive order."""
        return int(self._coefficients.shape[0])

    def _state_space(self) -> _LinearGaussianStateSpaceModel:
        """Assemble the observable form."""
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
        return _LinearGaussianStateSpaceModel(
            design,
            obs_cov,
            transition,
            selection,
            state_cov,
            state_intercept=state_intercept,
        )

    def filter(self, endog: npt.ArrayLike) -> _KalmanFilterResult:
        """Run the forward pass over the mixed-frequency sample.

        Args:
            endog: The ``(nobs, k)`` sample at the high frequency, ``nan``
                wherever a variable is not observed.

        Returns:
            The Kalman filter output over the stacked state.
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
            DimensionError: If the sample has no rows beyond the
                autoregressive order, leaving no innovation to compute.
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

        Args:
            endog: The mixed-frequency sample.

        Returns:
            Mapping from variable name to its inferred current value.
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
    """A fitted MIDAS vector autoregression.

    The reduced-form surface is inherited unchanged, and everything it reports
    is conditional in exactly one place: the coefficient standard errors, and
    every test built from them, take the estimated lag polynomial as known.
    :meth:`joint_stderr` is the honest alternative and says what it costs.

    :meth:`forecast` is conditional in the same sense a VARX forecast is: the
    model holds no process for the high-frequency block, so it asks for the
    future sub-period path and refuses without it.

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
    """

    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    exog_high: npt.NDArray[np.float64]
    exog: npt.NDArray[np.float64]
    exog_names: tuple[str, ...]
    midas_coefficients: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    period: int
    midas_lags: int

    @property
    def k_exog(self) -> int:
        """Number of high-frequency series."""
        return len(self.exog_names)

    @property
    def midas_weights(self) -> npt.NDArray[np.float64]:
        """The estimated weight curves, ``(midas_lags, m)``, most recent first."""
        return np.column_stack([_midas_weights(row, self.midas_lags) for row in self.theta])

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The compressed-regressor coefficients, in design order."""
        return (self.midas_coefficients.T,)

    def _trailing_labels(self) -> tuple[str, ...]:
        """Compressed-regressor column names."""
        return self.exog_names

    def forecast(
        self, steps: int = 1, *, exog_high_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Point forecasts conditional on a future high-frequency path.

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
        """Standard errors that also account for estimating the lag polynomial.

        The default surface -- ``bse``, ``tvalues``, ``conf_int``, and the
        coefficient table -- conditions on the estimated polynomial, treating
        the compressed regressor as data. Here the observed information of the
        covariance-concentrated Gaussian likelihood is computed numerically
        over the full parameter vector, every regression coefficient and every
        polynomial parameter jointly, and inverted. The profile-likelihood
        curvature is the correct marginal information for the retained
        parameters, so concentrating the covariance out changes nothing about
        the answer while removing ``k(k+1)/2`` dimensions from the Hessian.

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
        """Short specification label for a ranking table."""
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"MIDASVAR({self.order}, {self.midas_lags}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
    """A low-frequency VAR reading high-frequency data through a lag polynomial.

    The model is ``y_t = D d_t + A_1 y_{t-1} + ... + A_p y_{t-p} + B x_t(theta)
    + u_t`` at the low frequency, where ``x_t(theta)`` compresses the last
    ``midas_lags`` sub-period readings of each high-frequency series into one
    number with normalized exponential-Almon weights. Two parameters per series
    buy the whole within-period history, which is the entire point of MIDAS:
    the unrestricted alternative spends ``midas_lags`` coefficients per series
    per equation and eats the sample.

    **This class estimates**, unlike its state-space sibling, because nothing
    here is temporally aggregated: every regressor is observed, the model is
    linear conditional on the polynomial, and the likelihood identifies
    everything it touches. Estimation profiles the polynomial -- an inner
    least-squares solve conditional on ``theta``, an outer derivative-free
    search over it, multi-started because the decay direction can be
    multi-modal.

    Args:
        endog: The ``(nobs, k)`` low-frequency panel.
        exog_high: The ``(nobs * period, m)`` high-frequency panel,
            chronological, exactly ``period`` sub-period rows per low-frequency
            observation. Missing values are rejected: a ragged calendar is
            :class:`MFVAR`'s problem statement, not this model's.
        order: Endogenous autoregressive order.
        period: Sub-periods per low-frequency period, at least two -- with one
            sub-period nothing is mixed and
            :class:`~cultivars.var.reduced_form.VARX` is the right model.
        midas_lags: Sub-periods each weight curve spans. Defaults to
            ``period``; longer windows reach into earlier low-frequency
            periods and cost leading observations.
        trend: Deterministic terms.
        names: Endogenous labels. Defaults to ``y1 ... yk``.
        exog_names: High-frequency labels. Defaults to ``x1 ... xm``.

    Raises:
        SpecificationError: If ``period`` or ``midas_lags`` is malformed, or
            the labels are malformed or collide.
        DimensionError: If the two panels do not align at ``period`` rows per
            observation, or the sample is too short for the specification.

    Example:
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
        """Validate both panels and the specification."""
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
        """The validated high-frequency panel."""
        return self._exog_high

    @property
    def exog_names(self) -> tuple[str, ...]:
        """High-frequency labels."""
        return self._exog_names

    @property
    def k_exog(self) -> int:
        """Number of high-frequency series."""
        return int(self._exog_high.shape[1])

    @property
    def period(self) -> int:
        """Sub-periods per low-frequency period."""
        return self._period

    @property
    def midas_lags(self) -> int:
        """Sub-periods each weight curve spans."""
        return self._midas_lags

    @property
    def n_regressors(self) -> int:
        """Regressors per equation, including the compressed block."""
        return super().n_regressors + self.k_exog

    def _burn_for(self, order: int) -> int:
        """Leading observations lost to the lags or to the weight window."""
        reach = (self._midas_lags + self._period - 1) // self._period - 1
        return max(order, reach)

    def _max_supported_lags(self) -> int:
        """Largest endogenous order the sample can identify."""
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
        """Starting points: flat weights, gentle decay, and steep decay."""
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
