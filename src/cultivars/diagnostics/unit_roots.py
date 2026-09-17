# filepath: /src/cultivars/diagnostics/unit_roots.py
#
# [MIT header, Copyright (c) 2026 Nikhil Sunder]

"""Unit-root and stationarity tests: the question every model in the package asks first.

A level VAR, a cointegrating system, and a differenced ARMA are answers to
one question -- does the series carry a stochastic trend -- and the tests
here are how that question is put to the data. They come in two
families. The Dickey-Fuller line puts the unit root under the null:
:func:`adf` with its augmentation chosen on a common sample,
:func:`phillips_perron` with a kernel correction instead of augmentation,
:func:`dfgls` on a GLS-detrended series (Elliott, Rothenberg & Stock,
1996), and :func:`ng_perron`, whose ``M`` statistics keep size under a
large negative moving-average root that inflates the others. :func:`kpss`
puts stationarity under the null instead, and a series both families
fail to reject is one the sample cannot classify -- which the record
says rather than hides. :func:`zivot_andrews` allows one break at an
unknown date, because a broken trend is the classic way a stationary
series is mistaken for a unit root (Perron, 1989).

Every ``trend`` argument takes ``"n"`` (nothing), ``"c"`` (constant) or
``"ct"`` (constant and linear trend); KPSS, DF-GLS and Ng-Perron admit
only the last two.

References:
    Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
        for autoregressive time series with a unit root. *Journal of the
        American Statistical Association*, 74(366), 427-431.
    Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in
        autoregressive-moving average models of unknown order.
        *Biometrika*, 71(3), 599-607.
    Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root in
        time series regression. *Biometrika*, 75(2), 335-346.
    Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
        Testing the null hypothesis of stationarity against the
        alternative of a unit root. *Journal of Econometrics*, 54(1-3),
        159-178.
    Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). Efficient tests
        for an autoregressive unit root. *Econometrica*, 64(4), 813-836.
    Ng, S., & Perron, P. (2001). Lag length selection and the construction
        of unit root tests with good size and power. *Econometrica*,
        69(6), 1519-1554.
    Perron, P., & Qu, Z. (2007). A simple modification to improve the
        finite sample properties of Ng and Perron's unit root tests.
        *Economics Letters*, 94(1), 12-19.
    Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great
        crash, the oil-price shock, and the unit-root hypothesis.
        *Journal of Business & Economic Statistics*, 10(3), 251-270.
    MacKinnon, J. G. (1994). Approximate asymptotic distribution functions
        for unit-root and cointegration tests. *Journal of Business &
        Economic Statistics*, 12(2), 167-176.
    MacKinnon, J. G. (2010). Critical values for cointegration tests.
        Queen's Economics Department Working Paper 1227.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> walk = np.cumsum(rng.standard_normal(300))
    >>> adf(walk).reject(), kpss(walk).reject()
    (False, True)
    >>> noise = rng.standard_normal(300)
    >>> adf(noise).reject(), kpss(noise).reject()
    (True, False)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._core import (
    _DFGLS_CRITICAL,
    _GLS_DETREND_C,
    _KPSS_CRITICAL,
    _NG_PERRON_CRITICAL,
    _ZIVOT_ANDREWS_CRITICAL,
    _critical_value_table,
    _dickey_fuller_regression,
    _gls_detrend,
    _kpss_pvalue,
    _kpss_statistic,
    _long_run_variance,
    _mackinnon_critical_values,
    _mackinnon_pvalue,
    _newey_west_bandwidth,
    _ng_perron_statistics,
    _phillips_perron,
    _resolve_dickey_fuller_lags,
    _schwert_max_lags,
    _zivot_andrews,
    validate_choice,
    validate_endog,
)
from ..exceptions import SpecificationError
from . import UnitRootTest

__all__ = [
    "UnitRootTest",
    "adf",
    "dfgls",
    "kpss",
    "long_run_variance",
    "ng_perron",
    "phillips_perron",
    "zivot_andrews",
]


def adf(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "aic",
) -> UnitRootTest:
    """Augmented Dickey-Fuller test of a unit root.

    The augmentation absorbs serial correlation in the differences. When
    ``lags`` is not given it is chosen on the common sample that the
    largest candidate leaves, so the criteria compare like with like;
    the test itself is then run on the longest sample the chosen
    augmentation allows.

    Args:
        endog: The series.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Augmentation lags, or ``None`` to select.
        max_lags: Largest augmentation under selection; default
            Schwert's ``12 (T / 100)^(1/4)``.
        method: ``"aic"``, ``"bic"``, ``"t-stat"`` (general-to-specific at
            10%) or ``"maic"`` (Ng-Perron modified AIC).

    Returns:
        The :class:`UnitRootTest` with MacKinnon's response-surface
        p-value and finite-sample critical values.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable
            or the sample is too short.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.
    """
    validate_choice(trend, ("n", "c", "ct"), "trend")
    y = validate_endog(endog)
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    tau, _, _, _, n_eff = _dickey_fuller_regression(y, chosen, trend)
    return UnitRootTest(
        name="ADF",
        statistic=tau,
        pvalue=_mackinnon_pvalue(tau, trend),
        critical_values=_mackinnon_critical_values(trend, n_eff),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=chosen,
        nobs=n_eff,
        method=label,
    )


def phillips_perron(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    kernel: str = "bartlett",
    bandwidth: float | None = None,
) -> UnitRootTest:
    """Phillips-Perron ``Z_t`` test of a unit root.

    The un-augmented Dickey-Fuller ``t`` is corrected with a kernel
    estimate of the long-run variance of the residuals, so serial
    correlation is handled nonparametrically rather than by adding lags.
    The corrected statistic follows the same Dickey-Fuller law, and
    MacKinnon's p-value applies.

    Args:
        endog: The series.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for Newey-West's rule
            (Bartlett) or Andrews' plug-in (quadratic spectral).

    Returns:
        The :class:`UnitRootTest`; ``lags`` reports the bandwidth used.

    Raises:
        SpecificationError: If the trend or kernel is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.
    """
    validate_choice(trend, ("n", "c", "ct"), "trend")
    y = validate_endog(endog)
    z_t, _, n_eff = _phillips_perron(y, trend, kernel=kernel, bandwidth=bandwidth)
    width = _newey_west_bandwidth(n_eff) if bandwidth is None else int(np.floor(bandwidth))
    return UnitRootTest(
        name="Phillips-Perron",
        statistic=z_t,
        pvalue=_mackinnon_pvalue(z_t, trend),
        critical_values=_mackinnon_critical_values(trend, n_eff),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=width,
        nobs=n_eff,
        method=f"{kernel} kernel",
    )


def kpss(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    kernel: str = "bartlett",
    bandwidth: float | None = None,
) -> UnitRootTest:
    """KPSS test with stationarity under the null.

    The LM statistic is the scaled sum of squared partial sums of the
    residuals from a regression on the deterministic terms, divided by a
    kernel estimate of their long-run variance. The p-value is
    interpolated in the authors' table and clipped to ``[0.01, 0.10]``
    outside it, so ``0.01`` reads as "at most 1%" and ``0.10`` as "at
    least 10%".

    Args:
        endog: The series.
        trend: ``"c"`` (level stationarity) or ``"ct"`` (trend
            stationarity).
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for the automatic rule.

    Returns:
        The :class:`UnitRootTest` with ``null="stationarity"``.

    Raises:
        SpecificationError: If the trend or kernel is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite.
    """
    validate_choice(trend, ("c", "ct"), trend)
    y = validate_endog(endog)
    statistic, _ = _kpss_statistic(y, trend, kernel=kernel, bandwidth=bandwidth)
    table = dict(_KPSS_CRITICAL[trend])
    width = _newey_west_bandwidth(y.shape[0]) if bandwidth is None else int(np.floor(bandwidth))
    return UnitRootTest(
        name="KPSS",
        statistic=statistic,
        pvalue=_kpss_pvalue(statistic, trend),
        critical_values=_critical_value_table((table[0.01], table[0.05], table[0.10])),
        null="stationarity",
        lower_tail=False,
        trend=trend,
        lags=width,
        nobs=int(y.shape[0]),
        method=f"{kernel} kernel",
    )


def dfgls(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "maic",
) -> UnitRootTest:
    """Elliott-Rothenberg-Stock DF-GLS test of a unit root.

    The deterministic terms are removed by GLS under the local
    alternative ``1 + c / T`` (``c = -7`` with a constant, ``-13.5`` with
    a trend), and the Dickey-Fuller regression is run on the detrended
    series without deterministic terms. The result is close to the
    point-optimal test and far more powerful than ADF near the null.
    The augmentation is chosen on the OLS-detrended series, as Perron
    and Qu (2007) recommend: on GLS-detrended data the modified AIC
    over-selects badly whenever the series is far from the null. Even
    so, on a series close to white noise the modified AIC still
    selects long augmentations and the statistic loses power;
    ``method="aic"`` or a fixed ``lags`` is the remedy there. Critical
    values are the authors' asymptotic ones; no response surface
    exists, so ``pvalue`` is ``None``.

    Args:
        endog: The series.
        trend: ``"c"`` or ``"ct"``.
        lags: Augmentation lags, or ``None`` to select.
        max_lags: Largest augmentation under selection.
        method: Selection rule; the modified AIC is the authors'
            companion recommendation (Ng & Perron, 2001).

    Returns:
        The :class:`UnitRootTest`.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.
    """
    validate_choice(trend, ("c", "ct"), "trend")
    y = validate_endog(endog)
    detrended = _gls_detrend(y, trend, _GLS_DETREND_C[trend])
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    tau, _, _, _, n_eff = _dickey_fuller_regression(detrended, chosen, "n")
    return UnitRootTest(
        name="DF-GLS",
        statistic=tau,
        pvalue=None,
        critical_values=_critical_value_table(_DFGLS_CRITICAL[trend]),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=chosen,
        nobs=n_eff,
        method=label,
    )


def ng_perron(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "maic",
) -> UnitRootTest:
    """Ng-Perron ``M`` tests of a unit root on a GLS-detrended series.

    ``MZ_t`` is the primary statistic; ``MZ_a``, ``MSB`` and ``MP_T``
    travel as companions. The autoregressive spectral density at
    frequency zero replaces the kernel estimate the Phillips-Perron
    tests use, and with the modified AIC choosing the augmentation the
    tests keep their size under a large negative moving-average root,
    where ADF and PP over-reject badly. The augmentation is chosen on
    the OLS-detrended series (Perron & Qu, 2007) and the statistics
    computed on the GLS-detrended one. Critical values are asymptotic;
    ``pvalue`` is ``None``.

    The modified AIC is built for the neighbourhood of the null. Far
    from it -- a series close to white noise -- its penalty is dominated
    by the level coefficient and it selects long augmentations that cost
    the ``M`` statistics most of their power; ``method="aic"`` or a
    fixed ``lags`` is the remedy when the series is plainly stationary,
    at the price of size under a negative moving-average root.

    Args:
        endog: The series.
        trend: ``"c"`` or ``"ct"``.
        lags: Augmentation for the spectral estimate, or ``None`` to
            select.
        max_lags: Largest augmentation under selection.
        method: Selection rule; default the modified AIC.

    Returns:
        The ``MZ_t`` :class:`UnitRootTest` with the other three as
        ``companions``.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.
    """
    validate_choice(trend, ("c", "ct"), "trend")
    y = validate_endog(endog)
    detrended = _gls_detrend(y, trend, _GLS_DETREND_C[trend])
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    stats = _ng_perron_statistics(detrended, chosen, trend)
    table = _NG_PERRON_CRITICAL[trend]
    n_eff = int(y.shape[0] - 1 - chosen)

    def record(key: str, display: str, companions: tuple[UnitRootTest, ...]) -> UnitRootTest:
        return UnitRootTest(
            name=display,
            statistic=stats[key],
            pvalue=None,
            critical_values=_critical_value_table(table[key]),
            null="unit root",
            lower_tail=True,
            trend=trend,
            lags=chosen,
            nobs=n_eff,
            method=label,
            companions=companions,
        )

    companions = (record("MZa", "MZ_a", ()), record("MSB", "MSB", ()), record("MPT", "MP_T", ()))
    return record("MZt", "Ng-Perron MZ_t", companions)


def zivot_andrews(
    endog: npt.ArrayLike,
    *,
    model: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    trimming: float = 0.15,
) -> UnitRootTest:
    """Zivot-Andrews test of a unit root against a one-time break at an unknown date.

    For every candidate date inside the trimmed window the Dickey-Fuller
    regression is run with a constant, a trend, and the break dummies of
    the chosen model; the statistic is the smallest ``t`` on the level
    lag over the window, and the date that produces it is the estimated
    break. The null is a unit root *without* a break, so a rejection
    says "stationary around a broken trend", not "broken unit root".
    Critical values are the authors' asymptotic ones; ``pvalue`` is
    ``None``.

    Args:
        endog: The series.
        model: ``"c"`` for a break in the intercept, ``"t"`` in the trend
            slope, ``"ct"`` in both.
        lags: Fixed augmentation, or ``None`` to select once by the
            t-statistic rule on the no-break regression.
        max_lags: Largest augmentation under selection.
        trimming: Fraction of the sample excluded at each end.

    Returns:
        The :class:`UnitRootTest` with ``break_index`` set.

    Raises:
        SpecificationError: If the model, trimming, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or every candidate
            is singular.
    """
    validate_choice(model, ("c", "t", "ct"), "model")
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    y = validate_endog(endog)
    ceiling = _schwert_max_lags(y.shape[0]) if max_lags is None else int(max_lags)
    statistic, break_index, used = _zivot_andrews(
        y, model=model, lags=lags, max_lags=ceiling, trimming=trimming
    )
    return UnitRootTest(
        name="Zivot-Andrews",
        statistic=statistic,
        pvalue=None,
        critical_values=_critical_value_table(_ZIVOT_ANDREWS_CRITICAL[model]),
        null="unit root",
        lower_tail=True,
        trend="ct",
        lags=used,
        nobs=int(y.shape[0] - 1 - used),
        method="fixed" if lags is not None else f"t-stat (max {ceiling})",
        break_index=break_index,
    )


def long_run_variance(
    endog: npt.ArrayLike, *, kernel: str = "bartlett", bandwidth: float | None = None
) -> float:
    """Kernel estimate of a series' long-run variance ``sum_j gamma_j`` about its mean.

    Args:
        endog: The series.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for the automatic rule.

    Returns:
        The estimate.

    Raises:
        SpecificationError: If the kernel or bandwidth is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(2000)
        >>> bool(0.8 < long_run_variance(e) < 1.2)
        True
    """
    y = validate_endog(endog)
    return _long_run_variance(y - y.mean(), kernel=kernel, bandwidth=bandwidth)
