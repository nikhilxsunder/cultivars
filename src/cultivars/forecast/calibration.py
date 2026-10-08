# filepath: /src/cultivars/forecast/calibration.py
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
r"""Calibration: does reality land where the densities said it would?

A density forecast can *score* well and still lie about its own
uncertainty -- too narrow in the tails, biased in the center -- and the
probability integral transform is how that shows. Feed each realization
through its own predictive distribution,

.. math::

   u_t = \hat F_t(y_t), \qquad t = 1, \dots, T,

and a calibrated forecaster produces independent uniforms on
:math:`(0, 1)`. The shape of the departure is the diagnosis. A U-shaped
PIT histogram means overconfidence (reality keeps landing in the tails
the forecaster ruled out), a hump means overdispersion, a slope means
bias. Berkowitz's (2001) test makes the reading formal on the
normal-quantile scale :math:`z_t = \Phi^{-1}(u_t)`, where uniformity
plus independence becomes the null

.. math::

   z_t = \mu + \rho\, z_{t-1} + \varepsilon_t,
   \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2),
   \qquad H_0 : \mu = 0,\ \rho = 0,\ \sigma^2 = 1,

a three-restriction Gaussian likelihood ratio with an exact
:math:`\chi^2_3` reference distribution.

Two commitments shape the surface. First, the input grammar differs
from scoring's on purpose: calibration is a statement across *many*
evaluation origins, so :class:`Calibration` takes the rolling stack --
one predictive sample per origin, one realization per origin -- that a
driver loop produces, and refuses fewer than ten origins rather than
fit a three-parameter likelihood to a handful of points. A
:class:`~cultivars.forecast.backtest.BacktestResult` supplies exactly
that stack for one horizon through its
:meth:`~cultivars.forecast.backtest.BacktestResult.record` method, so
the alignment is never the caller's arithmetic. Second, one caveat is
stated rather than hidden: the Berkowitz null assumes the transforms
are one-step objects; multi-step forecasts overlap mechanically, and
their PIT series carry serial correlation the test will read as
miscalibration. The record says so in its summary notes, and the
formal verdict is best read at horizon one.

Layout. :class:`Calibration` is the producer and
:class:`CalibrationResult` the record, which carries the ``(T, k)`` PIT
series, one Berkowitz
:class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest` per
series, and the :meth:`~CalibrationResult.histogram` that draws the
picture. The numerics live in ``_core``: ``pit_from_draws`` evaluates
the predictive sample's empirical distribution function at the outcome
with the half-count tie convention and a half-draw clip away from the
endpoints, and ``_berkowitz_likelihood_ratio`` fits the AR(1) by
conditional maximum likelihood and returns the statistic, its degrees
of freedom, and the p-value.

References:
    Berkowitz, J. (2001). Testing density forecasts, with applications to
    risk management. *Journal of Business & Economic Statistics*, 19(4),
    465-474.

    Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating
    density forecasts with applications to financial risk management.
    *International Economic Review*, 39(4), 863-883.

    Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
    forecasts, calibration and sharpness. *Journal of the Royal
    Statistical Society: Series B*, 69(2), 243-268.

Example:
    A forecaster whose densities are half as wide as they should be,
    read first by the histogram and then by the test:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> paths = 0.5 * rng.standard_normal((80, 500))
    >>> outcomes = rng.standard_normal(80)
    >>> record = Calibration(paths, outcomes).compute()
    >>> record.histogram(4).ravel()
    array([37.,  8.,  9., 26.])
    >>> bool(record.berkowitz("y1").reject(alpha=0.05))
    True

    The one-step slice of a BVAR backtest, handed over already aligned:

    >>> from cultivars.forecast.backtest import Backtest
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> y = np.random.default_rng(0).standard_normal((160, 2))
    >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=100, seed=0)
    >>> backtest = Backtest(y, fit, horizons=2, start=120, step=2).run(seed=0)
    >>> Calibration(*backtest.record(1)).compute().pit.shape
    (20, 2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.stats as sst

from .._core import SummaryTable, _berkowitz_likelihood_ratio, pit_from_draws
from .._internals import _LikelihoodRatioTest, _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["Calibration", "CalibrationResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class CalibrationResult(_SummaryMixin):
    r"""The PIT series and its formal reading, per series.

    One probability integral transform per evaluation origin and series,

    .. math::

       u_{t,j} = F_{t,j}(y_{t,j})
       = \frac{1}{S} \sum_{s=1}^{S} \mathbb{1}\{x_{t,j}^{(s)} < y_{t,j}\}
       + \frac{1}{2S} \sum_{s=1}^{S} \mathbb{1}\{x_{t,j}^{(s)} = y_{t,j}\},

    the predictive sample's empirical distribution function at the
    outcome, and beside the series one Berkowitz verdict per column. If
    the predictive distributions were correct, the :math:`u_{t,j}` are
    independent uniforms on :math:`(0, 1)`; the departure's shape is the
    diagnosis, and :meth:`histogram` draws it. The verdict is Berkowitz's
    likelihood ratio on the normal quantiles :math:`z_{t} =
    \Phi^{-1}(u_{t})`, which under the null are independent standard
    normals: an AR(1) with free mean, slope, and variance is fitted and
    tested against zero mean, zero slope, and unit variance, a
    three-restriction :math:`\chi^2` test.

    Attributes:
        names: One label per series.
        pit: ``(T, k)`` probability integral transforms, one per
            evaluation origin, interior to ``(0, 1)``. Uniform under
            correct calibration.
        tests: One Berkowitz likelihood-ratio verdict per series, in
            ``names`` order.

    Note:
        The transforms are clipped half a draw's probability away from
        the endpoints, ``0.5 / S`` and ``1 - 0.5 / S``, so that the normal
        quantile the test needs is always finite; a finite sample cannot
        distinguish an outcome beyond every draw from one at the edge.
        The Berkowitz null assumes one-step transforms. Multi-step
        forecasts overlap mechanically, and the serial correlation their
        PIT series inherits reads as miscalibration here; hand the record
        one horizon at a time and read the multi-step verdicts as
        conservative.

    See Also:
        * :class:`Calibration` -- the producer.
        * :meth:`~cultivars.forecast.backtest.BacktestResult.record` --
          gives the ``(paths, realized)`` pair for one horizon that
          :class:`Calibration` takes.
        * :class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest`
          -- the record each entry of ``tests`` is.

    References:
        Berkowitz, J. (2001). Testing density forecasts, with applications
        to risk management. *Journal of Business & Economic Statistics*,
        19(4), 465-474.

        Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating
        density forecasts with applications to financial risk management.
        *International Economic Review*, 39(4), 863-883.

    Example:
        A calibrated forecaster, then one whose densities are half as
        wide as they should be, which the U-shaped histogram and the
        Berkowitz test both catch:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> paths = rng.standard_normal((80, 500, 2))
        >>> outcomes = rng.standard_normal((80, 2))
        >>> record = Calibration(paths, outcomes).compute()
        >>> record.n_origins, record.pit.shape
        (80, (80, 2))
        >>> round(float(record.pit[:, 0].mean()), 2), bool(record.berkowitz("y1").pvalue > 0.05)
        (0.48, True)
        >>> rng = np.random.default_rng(1)
        >>> narrow = Calibration(0.5 * rng.standard_normal((80, 500)), rng.standard_normal(80))
        >>> overconfident = narrow.compute()
        >>> overconfident.histogram(4).ravel()
        array([37.,  8.,  9., 26.])
        >>> bool(overconfident.berkowitz("y1").pvalue < 0.001)
        True
    """

    names: tuple[str, ...]
    """Series labels, in column order; ``"y1"``, ``"y2"``, ... unless the producer was given names.

    A single series is ``"y1"``, not ``"y"``.
    """
    pit: npt.NDArray[np.float64] = field(repr=False)
    """``(T, k)`` probability integral transforms, one row per origin.

    Each is the predictive sample's empirical distribution function at
    the outcome with the half-count tie convention, clipped to
    ``[0.5 / S, 1 - 0.5 / S]``. Kept out of the repr.
    """
    tests: tuple[_LikelihoodRatioTest, ...]
    """One Berkowitz likelihood-ratio record per series, ``df = 3``, in ``names`` order."""

    @property
    def n_origins(self) -> int:
        """Evaluation origins, :math:`T`, the length of each PIT series.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> record = Calibration(rng.standard_normal((40, 200)), rng.standard_normal(40))
            >>> record.compute().n_origins
            40
        """
        return int(self.pit.shape[0])

    def _index(self, name: str) -> int:
        """The column index of a series label, after checking it exists.

        Args:
            name: One of ``names``.

        Returns:
            Its position.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcomes = rng.standard_normal((40, 200, 2)), rng.standard_normal((40, 2))
            >>> record = Calibration(paths, outcomes)
            >>> record.compute()._index("y2")
            1
            >>> record.compute()._index("z")
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown series 'z'; the panel has ('y1', 'y2').
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown series {name!r}; the panel has {self.names}."
            ) from None

    def berkowitz(self, name: str) -> _LikelihoodRatioTest:
        """One series' Berkowitz verdict.

        The likelihood-ratio record for that column of ``tests``, with
        ``statistic``, ``df = 3``, ``pvalue``, and ``reject(alpha=...)``.

        Args:
            name: A series label.

        Returns:
            The likelihood-ratio test record.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> record = Calibration(rng.standard_normal((80, 500)), rng.standard_normal(80))
            >>> verdict = record.compute().berkowitz("y1")
            >>> verdict.df, bool(verdict.reject(alpha=0.05))
            (3, False)
        """
        return self.tests[self._index(name)]

    def histogram(self, bins: int = 10) -> npt.NDArray[np.float64]:
        """PIT histogram counts on equal-width bins over ``(0, 1)``.

        The picture the diagnosis reads: flat is calibrated, U-shaped is
        overconfident, hump-shaped is overdispersed, sloped is biased.
        Each column sums to ``n_origins``, and under correct calibration
        each cell has expectation ``n_origins / bins``.

        Args:
            bins: Number of equal-width bins, at least 2.

        Returns:
            A ``(bins, k)`` array of counts.

        Raises:
            SpecificationError: If fewer than two bins are asked for.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcomes = rng.standard_normal((80, 500, 2)), rng.standard_normal((80, 2))
            >>> record = Calibration(paths, outcomes)
            >>> counts = record.compute().histogram(4)
            >>> counts.shape, counts.sum(axis=0)
            ((4, 2), array([80., 80.]))
            >>> record.compute().histogram(1)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: bins must be at least 2; got 1.
        """
        if bins < 2:
            raise SpecificationError(f"bins must be at least 2; got {bins}.")
        edges = np.linspace(0.0, 1.0, bins + 1)
        out = np.empty((bins, len(self.names)))
        for index in range(len(self.names)):
            out[:, index] = np.histogram(self.pit[:, index], bins=edges)[0]
        return out

    def _summary_table(self) -> SummaryTable:
        r"""Build the structured summary: one row per series.

        The PIT mean and standard deviation against their uniform
        benchmarks of 0.5 and :math:`1/\sqrt{12} \approx 0.289`, and
        the Berkowitz p-value; the notes give the reading of each
        departure and restate the one-step caveat.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcomes = rng.standard_normal((80, 500, 2)), rng.standard_normal((80, 2))
            >>> record = Calibration(paths, outcomes)
            >>> table = record.compute()._summary_table()
            >>> table.columns, table.rows[0]
            (('series', 'PIT mean', 'PIT sd', 'Berkowitz p'), ('y1', '0.477', '0.285', '0.9890'))
        """
        rows = tuple(
            (
                name,
                f"{float(self.pit[:, index].mean()):.3f}",
                f"{float(self.pit[:, index].std(ddof=0)):.3f}",
                f"{self.tests[index].pvalue:.4f}",
            )
            for index, name in enumerate(self.names)
        )
        notes = [
            "Under correct calibration the PIT series is uniform: mean "
            "0.500, standard deviation 0.289, Berkowitz p-value well away "
            "from zero. A U-shaped histogram is overconfidence, a hump is "
            "overdispersion, a slope is bias -- histogram() draws the "
            "picture.",
            "The Berkowitz null assumes one-step transforms; multi-step "
            "forecasts overlap mechanically, and their serial correlation "
            "reads as miscalibration here.",
        ]
        return SummaryTable(
            title="Density Forecast Calibration",
            metadata=(
                ("Series", f"{len(self.names)}"),
                ("Origins", f"{self.n_origins}"),
            ),
            columns=("series", "PIT mean", "PIT sd", "Berkowitz p"),
            rows=rows,
            notes=tuple(notes),
        )


class Calibration:
    r"""Read a rolling forecast record's calibration.

    Takes the stack a driver loop produces -- one predictive sample per
    evaluation origin beside the outcome that origin was forecasting --
    and asks whether reality landed where the densities said it would.
    Each outcome is passed through its own predictive distribution,

    .. math::

       u_t = \hat F_t(y_t),

    and under correct calibration the :math:`u_t` are independent
    uniforms. :meth:`compute` records that series and, on its normal
    quantiles :math:`z_t = \Phi^{-1}(u_t)`, runs Berkowitz's (2001)
    likelihood-ratio test of zero mean, zero slope, and unit variance
    against a free Gaussian AR(1), one verdict per series.

    Args:
        paths: ``(T, n_draws, k)`` predictive samples, one block per
            evaluation origin; ``(T, n_draws)`` is promoted to ``k = 1``.
        realized: ``(T, k)`` outcomes, aligned origin by origin; ``(T,)``
            is promoted to ``k = 1``.
        names: One label per series. Defaults to ``y1 ... yk``.

    Attributes:
        _paths: The validated ``(T, n_draws, k)`` predictive stack.
        _realized: The validated ``(T, k)`` outcomes.
        _names: The resolved series labels.

    Raises:
        DimensionError: If the stacks cannot be aligned, or the labels do
            not match the series.
        SpecificationError: If the record is too short for the Berkowitz
            null to mean anything.
        NumericalError: If any input is not finite.

    Note:
        The input grammar differs from scoring's on purpose: calibration
        is a statement across *many* origins, so the class takes the
        rolling stack rather than one predictive sample, and refuses
        fewer than ten origins, below which the three-parameter Berkowitz
        likelihood has nothing to estimate from. The alignment of
        ``paths[t]`` with ``realized[t]`` is the caller's promise; a
        :class:`~cultivars.forecast.backtest.BacktestResult` keeps it by
        construction and hands over one horizon's pair through
        :meth:`~cultivars.forecast.backtest.BacktestResult.record`.

    Warning:
        The Berkowitz null assumes one-step transforms. A multi-step
        horizon's PIT series is serially correlated by construction,
        which the test reads as miscalibration; read its verdicts as
        conservative or restrict the formal test to horizon one.

    See Also:
        * :class:`CalibrationResult` -- the record :meth:`compute`
          returns.
        * :class:`~cultivars.forecast.backtest.Backtest` -- produces the
          aligned stack from a fitting function.
        * :class:`~cultivars.forecast.scoring.DensityScore` -- the
          sharpness side of the same question, scored one origin at a
          time.

    References:
        Berkowitz, J. (2001). Testing density forecasts, with applications
        to risk management. *Journal of Business & Economic Statistics*,
        19(4), 465-474.

        Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating
        density forecasts with applications to financial risk management.
        *International Economic Review*, 39(4), 863-883.

    Example:
        A synthetic calibrated stack, then the one-step slice of a BVAR
        backtest fed straight from its record:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> paths = rng.standard_normal((80, 500, 2))
        >>> outcomes = rng.standard_normal((80, 2))
        >>> record = Calibration(paths, outcomes).compute()
        >>> record.pit.shape
        (80, 2)
        >>> bool(record.berkowitz("y1").pvalue > 0.01)
        True
        >>> from cultivars.forecast.backtest import Backtest
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> y = np.random.default_rng(0).standard_normal((160, 2))
        >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=100, seed=0)
        >>> backtest = Backtest(y, fit, horizons=2, start=120, step=2).run(seed=0)
        >>> Calibration(*backtest.record(1)).compute().pit.shape
        (20, 2)
    """

    __slots__ = ("_names", "_paths", "_realized")

    def __init__(
        self,
        paths: npt.ArrayLike,
        realized: npt.ArrayLike,
        *,
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate and align the rolling stacks.

        Both inputs are coerced to ``float64`` and a single series
        promoted to ``k = 1``; the origin and series counts must then
        agree, at least ten origins must be present, and every value
        must be finite. Labels default to ``y1 ... yk`` and are otherwise
        checked one per series.

        Args:
            paths: ``(T, n_draws, k)`` or ``(T, n_draws)`` predictive
                samples.
            realized: ``(T, k)`` or ``(T,)`` outcomes.
            names: Series labels, or ``None`` for the default.

        Raises:
            DimensionError: If the shapes do not align or the labels do
                not match the series.
            SpecificationError: If fewer than ten origins are present.
            NumericalError: If any input is not finite.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcomes = rng.standard_normal((80, 100, 2)), rng.standard_normal((70, 2))
            >>> Calibration(paths, outcomes)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: paths must be (T, n_draws, k) against (T, k) ...
            >>> short = rng.standard_normal((5, 100)), rng.standard_normal(5)
            >>> Calibration(*short)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a calibration record over 5 origins cannot ...
        """
        block = np.asarray(paths, dtype=np.float64)
        if block.ndim == 2:
            block = block[:, :, None]
        outcome = np.asarray(realized, dtype=np.float64)
        if outcome.ndim == 1:
            outcome = outcome[:, None]
        if (
            block.ndim != 3
            or outcome.ndim != 2
            or block.shape[0] != outcome.shape[0]
            or block.shape[2] != outcome.shape[1]
        ):
            raise DimensionError(
                f"paths must be (T, n_draws, k) against (T, k) realizations; "
                f"got {np.asarray(paths).shape} against "
                f"{np.asarray(realized).shape}."
            )
        if block.shape[0] < 10:
            raise SpecificationError(
                f"a calibration record over {block.shape[0]} origins cannot "
                "support the Berkowitz test; provide at least 10."
            )
        if not (np.all(np.isfinite(block)) and np.all(np.isfinite(outcome))):
            raise NumericalError("paths and realizations must be finite.")
        k = block.shape[2]
        if names is None:
            resolved = tuple(f"y{index + 1}" for index in range(k))
        else:
            resolved = tuple(str(name) for name in names)
            if len(resolved) != k:
                raise DimensionError(
                    f"names must have one entry per series ({k}); got {len(resolved)}."
                )
        self._paths = block
        self._realized = outcome
        self._names = resolved

    def compute(self) -> CalibrationResult:
        r"""Transform every origin and test every series.

        Origin by origin, the outcome is passed through the empirical
        distribution function of its predictive sample
        (``pit_from_draws``, with the half-count tie convention and the
        half-draw clip); the whole ``(T, k)`` series is then mapped to
        normal quantiles and each column handed to
        ``_berkowitz_likelihood_ratio``, which returns the statistic, its
        three degrees of freedom, and the :math:`\chi^2_3` p-value that
        become one :class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest`
        per series. Deterministic: nothing here draws.

        Returns:
            The :class:`CalibrationResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcomes = rng.standard_normal((80, 500, 2)), rng.standard_normal((80, 2))
            >>> record = Calibration(paths, outcomes, names=("gdp", "cpi")).compute()
            >>> record.names, len(record.tests), record.tests[0].df
            (('gdp', 'cpi'), 2, 3)
            >>> bool(np.all((record.pit > 0) & (record.pit < 1)))
            True
        """
        origins, _, k = self._paths.shape
        pit = np.empty((origins, k))
        for origin in range(origins):
            pit[origin] = pit_from_draws(self._paths[origin], self._realized[origin])
        transformed = np.asarray(sst.norm.ppf(pit), dtype=np.float64)
        tests = tuple(
            _LikelihoodRatioTest(statistic=stat, df=df, pvalue=p)
            for stat, df, p in (
                _berkowitz_likelihood_ratio(transformed[:, index]) for index in range(k)
            )
        )
        return CalibrationResult(names=self._names, pit=pit, tests=tests)
