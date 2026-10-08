# filepath: /src/cultivars/multivariate/structural/external_instruments.py
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
r"""External-instrument identification: one shock, named from outside the system.

A reduced-form VAR pins down the innovation covariance :math:`\Sigma_u` and
nothing about the matrix :math:`B` in :math:`u_t = B \varepsilon_t`. An
instrument :math:`z_t` with

.. math::

   \mathbb{E}[z_t \varepsilon_{1t}] \neq 0,
   \qquad
   \mathbb{E}[z_t \varepsilon_{jt}] = 0 \quad (j \neq 1),

gives :math:`\mathbb{E}[z_t u_t] \propto b_1`: the column of :math:`B`
belonging to the shock the instrument speaks for, up to a scale that the
unit-variance normalization :math:`b_1' \Sigma_u^{-1} b_1 = 1` fixes.
Narrative tax changes, high-frequency monetary surprises, oil supply
disruptions: the instrument brings information the reduced form never had,
and in exchange identifies only that one column. The partial result is the
honest one.

Two commitments shape the surface. First, what is not identified is not
fabricated. The result carries one column, and the shared
:class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
surface is built so a single column supports impulse responses, variance
shares, shock recovery and historical contributions without inventing the
columns it does not have; the variance shares therefore need not sum to one,
and the summary says which shocks remain unidentified. Second, the two
instrument conditions are kept apart. Relevance is a testable implication
and the first-stage F statistic reports it; exogeneity is an assumption that
no statistic in the system can check, and the result's restriction note
says so on every summary rather than letting a large F stand in for both.

Layout. :class:`ProxySVAR` is an ``_IdentificationModel`` from
``_internals``, which validates the closed-system contract of the source
result and exposes its names; the instrument panel is validated and
demeaned here, ``identify()`` runs the two-stage least squares on the
source's residuals, and the column is packaged into ``SVARResult`` from
:mod:`~cultivars.multivariate.structural.zero_restrictions` with
``scheme="proxy"`` and the first-stage diagnostics. Full identification from
an ordering is :mod:`~cultivars.multivariate.structural.zero_restrictions`;
set identification without an instrument is
:mod:`~cultivars.multivariate.structural.sign_restrictions`.

References:
    Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and
    corporate income tax changes in the United States. *American Economic
    Review*, 103(4), 1212-1247.

    Stock, J. H., & Watson, M. W. (2018). Identification and estimation of
    dynamic causal effects in macroeconomics using external instruments.
    *Economic Journal*, 128(610), 917-948.

    Montiel Olea, J. L., Stock, J. H., & Watson, M. W. (2021). Inference in
    structural vector autoregressions identified with an external
    instrument. *Journal of Econometrics*, 225(1), 74-87.

Example:
    A bivariate system whose first shock moves both variables on impact and
    a noisy proxy for it. The column is recovered, the variance shares of
    the one identified shock stay below one for the variable it only partly
    drives, and the historical decomposition carries one column:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
    >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
    >>> eps = rng.standard_normal((402, 2))
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = A @ y[t - 1] + B @ eps[t]
    >>> proxy = eps[:, 0] + 0.5 * rng.standard_normal(402)
    >>> res = VAR(y, order=1, names=("r", "x")).fit()
    >>> svar = ProxySVAR(res, proxy[1:], shock="policy").identify()
    >>> svar.impact.round(2).ravel(), svar.is_complete
    (array([1. , 0.5]), False)
    >>> shares = svar.fevd(10)[-1, :, 0]
    >>> bool(shares[0] > 0.95), bool(0.1 < shares[1] < 0.4)
    (True, True)
    >>> svar.historical_decomposition().shape
    (401, 2, 1)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..._core import ClosedSystemResult
from ..._internals import _IdentificationModel
from ...exceptions import DimensionError, NumericalError, SpecificationError
from .zero_restrictions import SVARResult

__all__ = ["ProxySVAR"]


class ProxySVAR(_IdentificationModel[SVARResult]):
    r"""External-instrument identification, Mertens-Ravn (2013).

    An instrument :math:`z_t` that is correlated with one structural shock
    and uncorrelated with the others,

    .. math::

       \mathbb{E}[z_t \varepsilon_{1t}] = \alpha \neq 0,
       \qquad
       \mathbb{E}[z_t \varepsilon_{jt}] = 0 \quad (j \neq 1),

    identifies that shock's impact column :math:`b_1` up to scale, because
    :math:`\mathbb{E}[z_t u_t] = \alpha\, b_1` for the reduced-form
    innovations :math:`u_t = B \varepsilon_t`. The column is recovered by
    two-stage least squares on the innovations: the instrument's covariance
    with each innovation is proportional to the column, the ratio to the
    normalization variable's covariance removes the unknown :math:`\alpha`,
    and the unit-variance rescaling :math:`b_1' \Sigma_u^{-1} b_1 = 1` pins
    the scale against the innovation covariance. Nothing else is identified,
    and the result carries exactly that one column.

    Note:
        Both instrument conditions are assumptions, not testable
        implications. Relevance shows up in the first-stage F statistic the
        summary reports, and a weak first stage (F in the single digits)
        means the column is poorly determined and its sign uncertain -- the
        statistic is the classical homoskedastic F, which overstates
        strength under heteroskedasticity. Exogeneity never shows up
        anywhere and must be argued from how the instrument was built: a
        narrative series, a high-frequency surprise, a supply disruption.

    Attributes:
        _source: The closed reduced-form result being identified.
        _instruments: The ``(nobs, m)`` demeaned instrument panel.
        _pivot: Column index of the normalization variable.
        _shock: The label of the identified shock.

    See Also:
        * :class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
          -- the partially identified result returned.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
          -- full identification from a Cholesky ordering, with no outside
          information.
        * :mod:`~cultivars.multivariate.structural.sign_restrictions` -- set
          identification when no instrument exists.

    References:
        Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal
        and corporate income tax changes in the United States. *American
        Economic Review*, 103(4), 1212-1247.

        Stock, J. H., & Watson, M. W. (2018). Identification and estimation
        of dynamic causal effects in macroeconomics using external
        instruments. *Economic Journal*, 128(610), 917-948.

        Montiel Olea, J. L., Stock, J. H., & Watson, M. W. (2021). Inference
        in structural vector autoregressions identified with an external
        instrument. *Journal of Econometrics*, 225(1), 74-87.

    Example:
        A bivariate system whose first structural shock moves both variables
        on impact, with a noisy proxy for that shock. The identified column
        recovers the true impact to two decimals, the recovered shock series
        tracks the truth, and the column satisfies the unit-variance
        normalization against the innovation covariance:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
        >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> eps = rng.standard_normal((402, 2))
        >>> y = np.zeros((402, 2))
        >>> for t in range(1, 402):
        ...     y[t] = A @ y[t - 1] + B @ eps[t]
        >>> proxy = eps[:, 0] + 0.5 * rng.standard_normal(402)
        >>> res = VAR(y, order=1, names=("r", "x")).fit()
        >>> svar = ProxySVAR(res, proxy[1:], shock="policy").identify()
        >>> svar.scheme, svar.shock_names, svar.is_complete
        ('proxy', ('policy',), False)
        >>> svar.impact.round(2)
        array([[1. ],
               [0.5]])
        >>> bool(np.corrcoef(svar.structural_shocks()[:, 0], eps[1:, 0])[0, 1] > 0.99)
        True
        >>> column = svar.impact[:, 0]
        >>> round(float(column @ np.linalg.solve(res.sigma_u, column)), 6)
        1.0
    """

    __slots__ = ("_instruments", "_pivot", "_shock")

    def __init__(
        self,
        result: ClosedSystemResult,
        instruments: npt.ArrayLike,
        *,
        shock: str = "proxied",
        normalize: str | None = None,
    ) -> None:
        """Validate the source system, the instrument panel, and the anchor.

        Args:
            result: The fitted closed reduced-form result to identify.
            instruments: ``(nobs, m)`` instrument panel aligned with the
                *effective* sample -- one row per residual row of the result,
                which is the estimation sample minus the burned lags. A 1-D
                array is one instrument.
            shock: Label for the identified shock.
            normalize: Variable whose impact response is normalized positive,
                and whose innovation anchors the first stage. Defaults to the
                first variable.

        Raises:
            SpecificationError: If the result is not a closed system, or the
                normalization variable is unknown.
            DimensionError: If the instrument panel does not align with the
                effective sample.
            NumericalError: If the instrument panel is non-finite.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((120, 2))
            >>> res = VAR(y, order=2, names=("r", "x")).fit()
            >>> model = ProxySVAR(res, rng.standard_normal(118), normalize="x")
            >>> model.k_instruments, model._pivot, model._shock
            (1, 1, 'proxied')
            >>> ProxySVAR(res, rng.standard_normal(120))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: instruments must have one row per residual row ...
            >>> ProxySVAR(res, np.full(118, np.nan))
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: instruments must be finite.
            >>> ProxySVAR(res, rng.standard_normal(118), normalize="q")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown normalization variable 'q'; ...
        """
        super().__init__(result)
        n = int(np.asarray(result.resid).shape[0])
        z = np.asarray(instruments, dtype=np.float64)
        if z.ndim == 1:
            z = z[:, np.newaxis]
        if z.ndim != 2 or z.shape[0] != n:
            raise DimensionError(
                f"instruments must have one row per residual row ({n}), the "
                f"effective sample after the burned lags; got shape {z.shape}."
            )
        if not np.all(np.isfinite(z)):
            raise NumericalError("instruments must be finite.")
        self._instruments = z - z.mean(axis=0)
        anchor = self.names[0] if normalize is None else str(normalize)
        if anchor not in self.names:
            raise SpecificationError(
                f"unknown normalization variable {anchor!r}; expected one of {self.names}."
            )
        self._pivot = self.names.index(anchor)
        self._shock = str(shock)

    @property
    def k_instruments(self) -> int:
        """Number of instrument series.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((120, 2)), order=1).fit()
            >>> ProxySVAR(res, rng.standard_normal((119, 3))).k_instruments
            3
        """
        return int(self._instruments.shape[1])

    def identify(self) -> SVARResult:
        """Recover the proxied shock's impact column.

        The first stage regresses the normalization innovation on the
        instruments; the fitted values stand in for the shock, each
        innovation's covariance with them gives the column relative to the
        anchor, and the column is rescaled to unit shock variance. The anchor
        entry is positive by construction.

        Returns:
            A structural result carrying the one identified column, with the
            first-stage F statistic, the instrument count and the
            normalization in its diagnostics.

        Raises:
            NumericalError: If the instrument has no first-stage relationship
                with the normalization innovation, or there are no more
                residual rows than instruments.

        Example:
            The anchor's sign is fixed whatever the instrument's sign, so a
            proxy and its negative identify the same column; an unrelated
            series still produces a column, but a weak first stage says it
            is not to be trusted:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
            >>> eps = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.4 * y[t - 1] + B @ eps[t]
            >>> proxy = eps[:, 0] + 0.5 * rng.standard_normal(302)
            >>> res = VAR(y, order=1, names=("r", "x")).fit()
            >>> plus, minus = ProxySVAR(res, proxy[1:]), ProxySVAR(res, -proxy[1:])
            >>> bool(np.allclose(plus.identify().impact, minus.identify().impact))
            True
            >>> strong = dict(plus.identify().diagnostics)
            >>> strong["Normalization"], float(strong["First-stage F"]) > 100
            ('r > 0 on impact', True)
            >>> weak = ProxySVAR(res, rng.standard_normal(301)).identify()
            >>> float(dict(weak.diagnostics)["First-stage F"]) < 10
            True
            >>> ProxySVAR(res, np.ones(301)).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: the instrument has no first-stage relationship ...
        """
        resid = np.asarray(self.source.resid, dtype=np.float64)
        n = resid.shape[0]
        z = self._instruments
        m = self.k_instruments
        anchor = self.names[self._pivot]

        beta: npt.NDArray[np.float64] = np.linalg.lstsq(z, resid[:, self._pivot], rcond=None)[0]
        fitted = z @ beta
        explained = float(fitted @ fitted)
        residual = float(resid[:, self._pivot] @ resid[:, self._pivot]) - explained
        if explained <= 0.0 or n <= m:
            raise NumericalError(
                f"the instrument has no first-stage relationship with the "
                f"{anchor!r} innovation; a proxy that does not move the "
                "normalization variable identifies nothing."
            )
        f_stat = (explained / m) / (residual / (n - m))

        relative = resid.T @ fitted / explained
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        scale = float(relative @ np.linalg.solve(sigma, relative))
        column = (relative / np.sqrt(scale))[:, np.newaxis]

        return SVARResult(
            source=self.source,
            impact=column,
            shock_names=(self._shock,),
            scheme="proxy",
            restriction=(
                f"External instrument: {m} proxy series assumed correlated "
                f"with the {self._shock!r} shock and uncorrelated with every "
                "other structural shock. Relevance is reported below; "
                "exogeneity is an assumption the instrument's construction "
                "must defend, because no statistic here can."
            ),
            diagnostics=(
                ("First-stage F", f"{f_stat:.2f}"),
                ("Instruments", f"{m}"),
                ("Normalization", f"{anchor} > 0 on impact"),
            ),
        )
