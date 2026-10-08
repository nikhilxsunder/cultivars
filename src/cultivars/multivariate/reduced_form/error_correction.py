# filepath: /src/cultivars/multivariate/reduced_form/error_correction.py
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
r"""Vector error-correction models: the VAR in levels rewritten around its unit roots.

A VAR of order :math:`p` in :math:`k` integrated variables can be written
without loss as

.. math::

   \Delta y_t = \Pi\, y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i\, \Delta y_{t-i}
   + \mu_t + u_t,

and the content of cointegration is the rank of :math:`\Pi`. Rank
:math:`k` is a stationary system better fitted in levels; rank zero is a VAR
in differences with no long-run relations; rank :math:`r` strictly between
is the interesting case, :math:`\Pi = \alpha\beta'` with :math:`\beta`
spanning :math:`r` stationary combinations of non-stationary series,
:math:`\alpha` the speed at which each variable corrects them, and
:math:`k - r` common stochastic trends driving everything else. Johansen's
reduced-rank maximum likelihood estimates the pair by concentrating out the
short-run terms and solving an eigenvalue problem for :math:`\beta`; the
sequence of trace and maximum-eigenvalue tests chooses :math:`r` from the
same eigenvalues. The conditional variant of Pesaran, Shin and Smith models
only a block :math:`y` of the system given weakly exogenous integrated
regressors :math:`x` that share the cointegrating space but receive no
equations of their own.

Two commitments shape the surface. First, the results hold two coordinate
systems and are explicit about which member reads which. Inference --
``params``, ``bse``, ``pvalues``, ``conf_int``, the weak-exogeneity and
Granger tests -- reads the short-run regression that was actually run,
conditional on :math:`\beta`, which converges at rate :math:`T` and so can
be treated as known; propagation -- impulse responses, variance and
historical decompositions, forecasts -- reads the levels representation
:math:`A_1 = I + \Pi + \Gamma_1, \; A_i = \Gamma_i - \Gamma_{i-1}, \; A_p =
-\Gamma_{p-1}` the fit implies, so a VECM has no separate impulse-response
theory, only different coordinates, and the stability check permits exactly
the unit roots the rank specifies. Only the cointegrating *space* is
identified, so ``beta`` is printed on a readable normalization and carries
no standard errors, and the rank is chosen on the model by ``rank_test()``
rather than on the result by a likelihood ratio, whose distribution under a
rank restriction is not chi-squared. Second, the conditional model refuses
what it does not have. A :class:`VECMXResult` keeps every short-run member
and the conditional forecast, which needs only a path for :math:`x`, and
raises on the companion matrix, the stability check and every propagation
member, because a law of motion for :math:`x` was never written; what it
does offer is :meth:`VECMXResult.to_varx`, its own equations in levels,
which is the currency a global system links.

Layout. :class:`VECM` and :class:`VECMX` validate on
``_VectorErrorCorrectionModel`` and its exogenous extension in
``_internals``, which hold the Johansen case, the rank and the optional
exogenous block, map the case onto the unrestricted ``trend`` the short-run
regression sees through ``_UNRESTRICTED_TREND`` in ``_core``, run the
reduced-rank estimation in ``_fit_family`` -- concentration, eigenvalue
problem, then one multivariate regression on ``[deterministic | lagged
differences | error-correction terms]`` that reproduces Johansen's
:math:`\alpha` exactly by Frisch-Waugh-Lovell -- and expose ``rank_test()``
with simulated critical values for the closed and conditional nulls.
:class:`VECMResult` extends ``_ErrorCorrectionResult`` with the summary,
comparison, inference and propagation mixins, maps the Johansen case to the
levels ``trend`` through ``_LEVELS_TREND``, and overrides the coefficient
stack, the stability check, the forecast and the two tests;
:class:`VECMXResult` adds the exogenous block and the refusals, worded by
``_CONDITIONAL_REFUSAL`` in ``_core``, and packs its levels form into a
``_ConditionalLevels`` record for
:mod:`~cultivars.multivariate.reduced_form.closed_global`. The levels
system a VECM reparameterizes is
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`; the
rank test's public record is
:class:`~cultivars.diagnostics.cointegration.JohansenRankTest`, and the
unit-root tests that establish integration beforehand are in
:mod:`~cultivars.diagnostics.unit_roots`.

References:
    Johansen, S. (1991). Estimation and hypothesis testing of cointegration
    vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6),
    1551-1580.

    Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector
    Autoregressive Models*. Oxford University Press.

    Pesaran, M. H., Shin, Y., & Smith, R. J. (2000). Structural analysis of
    vector error correction models with exogenous I(1) variables. *Journal
    of Econometrics*, 97(2), 293-343.

    Harbo, I., Johansen, S., Nielsen, B., & Rahbek, A. (1998). Asymptotic
    inference on cointegrating rank in partial systems. *Journal of Business
    & Economic Statistics*, 16(4), 388-399.

    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer. Chapters 6-8.

Example:
    Three integrated series, two of them tied to the first. The rank test
    reads two relations off the eigenvalues, the fit at that rank leaves
    one common trend, the first series does not adjust to either relation,
    and the levels companion carries exactly the unit root the rank
    specifies:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> trend = np.cumsum(rng.standard_normal(400))
    >>> y = np.column_stack([
    ...     trend,
    ...     trend + rng.standard_normal(400),
    ...     0.5 * trend + rng.standard_normal(400),
    ... ])
    >>> model = VECM(y, order=2, rank=1)
    >>> rank = model.rank_test(simulations=2000).selected_rank()
    >>> rank
    2
    >>> res = VECM(y, order=2, rank=rank).fit()
    >>> res.n_common_trends, res.stability_check().n_unit_roots
    (1, 1)
    >>> bool(res.weak_exogeneity("y1").pvalue > 0.05)
    True
    >>> res.normalized_beta().round(1)[0]
    array([1., 1.])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from ..._core import (
    _CHOLESKY_NOTE,
    _CONDITIONAL_REFUSAL,
    _LEVELS_TREND,
    _UNRESTRICTED_TREND,
    _UNSTABLE_NOTE,
    CointegrationTrend,
    SummaryTable,
    deterministic_columns,
    validate_exog_matrix,
)
from ..._internals import (
    _ComparisonMixin,
    _ConditionalLevels,
    _ErrorCorrectionResult,
    _ExogenousVectorErrorCorrectionModel,
    _StabilityAssessment,
    _SummaryMixin,
    _VectorErrorCorrectionFit,
    _VectorErrorCorrectionModel,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
    _WaldTest,
)
from ...exceptions import DimensionError, SpecificationError

__all__ = ["VECM", "VECMX", "VECMResult", "VECMXResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VECMResult(
    _ErrorCorrectionResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted vector error-correction model.

    A VAR of order :math:`p` in :math:`k` integrated variables, rewritten
    around its unit roots,

    .. math::

       \Delta y_t = \Pi\, y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i\, \Delta y_{t-i}
       + \mu_t + u_t, \qquad \Pi = \alpha \beta',

    with :math:`\Pi` of rank :math:`r < k`: the columns of :math:`\beta` are
    the cointegrating vectors, :math:`\beta' y_{t-1}` the stationary
    disequilibria, :math:`\alpha` the speeds at which each variable corrects
    them, and :math:`k - r` common stochastic trends drive the rest.
    Estimation is Johansen's reduced-rank maximum likelihood: concentrate out
    the short-run terms, take :math:`\beta` from the eigenvalue problem, then
    regress the differences on ``[deterministic | lagged differences |
    error-correction terms]`` for everything else.

    The result carries two representations of one estimate, and each half of
    the inherited surface reads the one it needs.

    The *short-run* representation is what was actually regressed: differences
    on deterministic terms, lagged differences, and the error-correction terms.
    :attr:`design`, :attr:`resid`, :attr:`fittedvalues`, and everything built on
    the coefficient covariance -- ``params``, ``bse``, ``pvalues``,
    ``conf_int`` -- describe that regression, so a standard error here belongs
    to an ``alpha``, a ``Gamma``, or a deterministic term, conditional on
    ``beta``. The conditioning is what makes them exact rather than
    approximate: ``beta`` converges at rate ``T`` against the usual root-``T``,
    fast enough that everything estimated alongside it behaves as though the
    cointegrating space were known.

    The *levels* representation is the vector autoregression this model
    reparameterizes, recovered by :meth:`to_var` and exposed as
    :attr:`coefficients`. Impulse responses, the variance decomposition, and the
    historical decomposition read it, which is why they are inherited rather
    than rewritten: an error-correction model has no separate impulse-response
    theory, only different coordinates.

    Three members depart from the reduced-form surface deliberately.
    :meth:`stability_check` permits unit roots, because ``k - r`` of them are
    the specification rather than a failure. :meth:`forecast` folds any
    restricted deterministic term back out of the cointegrating space, since a
    constant that lives inside ``beta`` still shifts the level of a forecast.
    :meth:`granger_causality` restricts both transmission channels at once.

    Note:
        Two things this record does not do. It does not test its own rank:
        the Johansen trace and maximum-eigenvalue tests have non-standard
        null distributions and live on the model as ``rank_test()``, to be
        read before a rank is chosen, and a likelihood-ratio comparison of
        this result against a levels VAR or a VECM of another rank through
        the inherited ``likelihood_ratio_test`` is a chi-squared test of a
        restriction that is not chi-squared and should not be read. And it
        reports no standard errors for ``beta``: its asymptotics are
        mixed-Gaussian and conditional on the normalization, so the summary
        prints the cointegrating vectors without inference columns.

    Attributes:
        endog: The sample in levels.
        names: Variable labels, in Cholesky order.
        order: Lags of the levels system; ``order - 1`` lagged differences.
        rank: Cointegrating rank.
        cointegration_trend: The Johansen case the model was estimated under.
        trend: Deterministic terms of the *levels* representation, with any
            restricted term folded back out.
        alpha: ``(k, r)`` adjustment loadings.
        beta: ``(k or k + 1, r)`` cointegrating vectors.
        gamma: ``(p - 1, k, k)`` coefficients on lagged differences.
        coefficients: ``(p, k, k)`` implied levels autoregressive matrices.
        deterministic: ``(d, k)`` deterministic coefficients of the levels
            representation, restricted terms folded in.
        short_run_deterministic: ``(d_s, k)`` unrestricted deterministic terms,
            as they entered the short-run regression.
        eigenvalues: Squared canonical correlations, descending.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: Residuals of the short-run equation.
        fittedvalues: Fitted differences.
        design: The short-run regressor matrix.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Free parameters, covariance included.

    See Also:
        * :class:`VECM` -- the model whose ``fit()`` returns this record and
          whose ``rank_test()`` chooses the rank.
        * :class:`VECMXResult` -- the conditional extension with weakly
          exogenous integrated regressors.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the levels system this result reparameterizes and whose
          propagation surface it inherits.
        * :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
          -- the assembler that links conditional error-correction units
          into one closed system.

    References:
        Johansen, S. (1991). Estimation and hypothesis testing of
        cointegration vectors in Gaussian vector autoregressive models.
        *Econometrica*, 59(6), 1551-1580.

        Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated
        Vector Autoregressive Models*. Oxford University Press.

        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Chapters 6-7.

    Example:
        Two series sharing one stochastic trend, the second twice the first
        plus noise. At rank one the fit recovers the relation
        :math:`y_1 - y_2 / 2` with the second variable doing all the
        adjusting, the levels companion carries exactly one unit root, and
        the inherited propagation surface reads the levels representation:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> common = np.cumsum(rng.standard_normal(300))
        >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
        >>> res = VECM(y, order=2, rank=1).fit()
        >>> res.n_common_trends, res.trend
        (1, 'c')
        >>> res.normalized_beta().ravel().round(2)
        array([ 1.  , -0.49])
        >>> bool(res.pvalues["y1: ec1"] > 0.1), bool(res.pvalues["y2: ec1"] < 0.001)
        (True, True)
        >>> check = res.stability_check()
        >>> check.n_unit_roots, bool(abs(check.max_modulus - 1.0) < 1e-6)
        (1, True)
        >>> res.irf(4).shape, res.forecast(3).shape
        ((5, 2, 2), (3, 2))
    """

    coefficients: npt.NDArray[np.float64]
    r"""The ``(p, k, k)`` levels autoregressive matrices :math:`A_i` the fit implies.

    Built from the short-run estimate as :math:`A_1 = I + \Pi + \Gamma_1`,
    :math:`A_i = \Gamma_i - \Gamma_{i-1}` and :math:`A_p = -\Gamma_{p-1}`;
    what the propagation surface reads.
    """

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` deterministic coefficients of the levels representation.

    Any term restricted to the cointegrating space is folded back out through
    ``alpha``, so a restricted constant appears here as a row even though the
    short-run regression carried none.
    """

    @classmethod
    def _from_fit(
        cls, fit: _VectorErrorCorrectionFit, model: _VectorErrorCorrectionModel[Self]
    ) -> Self:
        """Assemble the public result from the internal fit and its model.

        Args:
            fit: The packed reduced-rank fit.
            model: The model that produced it, read for the sample, the
                labels, the order, the rank and the Johansen case.

        Returns:
            A populated result, with ``trend`` mapped from the Johansen case
            to the levels convention through ``_LEVELS_TREND``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(200))
            >>> y = np.column_stack([common, common + rng.standard_normal(200)])
            >>> model = VECM(y, order=2, rank=1, cointegration_trend="restricted_constant")
            >>> res = VECMResult._from_fit(model._fit_family(), model)
            >>> res.cointegration_trend, res.trend, res.beta.shape
            ('restricted_constant', 'c', (3, 1))
        """
        case = cast(CointegrationTrend, model.cointegration_trend)
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            rank=model.rank,
            cointegration_trend=case,
            trend=_LEVELS_TREND[case],
            alpha=fit.alpha,
            beta=fit.beta,
            gamma=fit.gamma,
            coefficients=fit.coefficients,
            deterministic=fit.deterministic,
            short_run_deterministic=fit.short_run_deterministic,
            eigenvalues=fit.eigenvalues,
            sigma_u=fit.sigma_u,
            sigma_ml=fit.sigma_ml,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            design=fit.design,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
        )

    # ------------------------------------------------------------- long run

    @property
    def cointegrating_matrix(self) -> npt.NDArray[np.float64]:
        r"""The long-run impact matrix :math:`\Pi = \alpha \beta'`, over the variables only.

        Any restricted deterministic row of ``beta`` is left out, so the
        result is ``(k, k)`` of rank :attr:`rank`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> res.cointegrating_matrix.shape, int(np.linalg.matrix_rank(res.cointegrating_matrix))
            ((2, 2), 1)
        """
        return self.alpha @ self.beta[: self.k_endog].T

    @property
    def n_common_trends(self) -> int:
        """Unit roots the specification carries, ``k - r``.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 3)), axis=0)
            >>> VECM(y, order=1, rank=1).fit().n_common_trends
            2
        """
        return self.k_endog - self.rank

    def normalized_beta(self, *, on: int = 0) -> npt.NDArray[np.float64]:
        """Cointegrating vectors scaled so one variable's loading is one in each.

        The eigenvectors come out of the decomposition normalized to make
        ``beta' S11 beta`` the identity, which is convenient for the algebra and
        unreadable as economics. Without further restrictions only the
        cointegrating *space* is identified, so any basis for it is as valid as
        any other; this picks the basis someone would write on a blackboard.

        Args:
            on: Index of the variable whose coefficient is set to one.

        Returns:
            A ``(k or k + 1, r)`` array, empty when the rank is zero.

        Raises:
            SpecificationError: If the index is out of range, or its loading is
                numerically zero in some vector, which means the normalization
                would divide by nothing and another variable should carry it.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> res.normalized_beta().ravel().round(2)
            array([ 1.  , -0.49])
            >>> res.normalized_beta(on=1).ravel().round(2)
            array([-2.02,  1.  ])
            >>> res.normalized_beta(on=2)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: on must index a variable, 0..1; got 2.
        """
        if not 0 <= on < self.k_endog:
            raise SpecificationError(f"on must index a variable, 0..{self.k_endog - 1}; got {on}.")
        if not self.rank:
            return np.zeros((self.beta.shape[0], 0), dtype=np.float64)
        pivots: npt.NDArray[np.float64] = self.beta[on]
        if np.any(np.abs(pivots) < 1e-10):
            raise SpecificationError(
                f"variable {self.names[on]!r} has a numerically zero loading in at least "
                "one cointegrating vector, so it cannot carry the normalization; choose a "
                "variable that enters every relation."
            )
        return np.asarray(self.beta / pivots, dtype=np.float64)

    def normalized_alpha(self, *, on: int = 0) -> npt.NDArray[np.float64]:
        """Adjustment loadings rescaled to pair with :meth:`normalized_beta`.

        Rescaling a cointegrating vector rescales its loading inversely, so the
        product and every test statistic are invariant and only the split
        between the two factors is convention. This returns the loadings that
        go with the readable basis.

        Args:
            on: Index of the variable whose coefficient normalizes each vector.

        Returns:
            A ``(k, r)`` array, empty when the rank is zero.

        Raises:
            SpecificationError: If the index is out of range or cannot carry
                the normalization.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> product = res.normalized_alpha() @ res.normalized_beta().T
            >>> bool(np.allclose(product, res.cointegrating_matrix))
            True
            >>> res.normalized_alpha().ravel().round(2)
            array([0.15, 2.63])
        """
        if not self.rank:
            return np.zeros((self.k_endog, 0), dtype=np.float64)
        return np.asarray(self.alpha * self.beta[on], dtype=np.float64)

    def error_correction_terms(self) -> npt.NDArray[np.float64]:
        r"""The fitted disequilibria, one column per cointegrating relation.

        The last ``rank`` columns of :attr:`design`: :math:`\hat\beta' y_{t-1}`
        on the decomposition's own normalization, including any restricted
        deterministic term.

        Returns:
            A ``(nobs, r)`` array, empty when the rank is zero.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> terms = res.error_correction_terms()
            >>> terms.shape, bool(np.allclose(terms[:, 0], y[1:-1] @ res.beta[:, 0]))
            ((298, 1), True)
        """
        return self.design[:, self.design.shape[1] - self.rank :]

    def to_var(self) -> npt.NDArray[np.float64]:
        """The levels autoregressive matrices this specification implies.

        Returns:
            A ``(p, k, k)`` stack from ``A_1 = I + Pi + Gamma_1``,
            ``A_i = Gamma_i - Gamma_{i-1}``, and ``A_p = -Gamma_{p-1}``. This is
            :attr:`coefficients`, named so that the conversion is discoverable
            from the econometrics rather than only from the attribute list.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> first, second = res.to_var()
            >>> bool(np.allclose(first, np.eye(2) + res.cointegrating_matrix + res.gamma[0]))
            True
            >>> bool(np.allclose(second, -res.gamma[0]))
            True
        """
        return self.coefficients

    # -------------------------------------------------- short-run coordinates

    def _deterministic_labels(self) -> tuple[str, ...]:
        """Unrestricted deterministic column names of the short-run equation.

        Returns:
            ``()``, ``("const",)`` or ``("const", "trend")`` according to the
            rows of :attr:`short_run_deterministic`; a restricted term is not
            a column here.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> VECM(y, order=2, rank=1).fit()._deterministic_labels()
            ('const',)
            >>> restricted = VECM(y, order=2, rank=1, cointegration_trend="restricted_constant")
            >>> restricted.fit()._deterministic_labels()
            ()
        """
        width = int(self.short_run_deterministic.shape[0])
        return ("const", "trend")[:width]

    def _lag_labels(self) -> tuple[str, ...]:
        """Lagged-difference column names.

        Returns:
            ``"D.<name>.L<lag>"`` for each of the ``order - 1`` lags, lag-major
            then variable-minor, matching the design columns.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> VECM(y, order=2, rank=1).fit()._lag_labels()
            ('D.y1.L1', 'D.y2.L1')
            >>> VECM(y, order=1, rank=1).fit()._lag_labels()
            ()
        """
        return tuple(
            f"D.{source}.L{lag + 1}" for lag in range(self.order - 1) for source in self.names
        )

    def _trailing_labels(self) -> tuple[str, ...]:
        """Error-correction column names.

        Returns:
            ``("ec1", ..., "ec<r>")``.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 3)), axis=0)
            >>> VECM(y, order=1, rank=2).fit()._trailing_labels()
            ('ec1', 'ec2')
        """
        return tuple(f"ec{i + 1}" for i in range(self.rank))

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The adjustment loadings, laid out as design rows.

        Returns:
            ``(alpha.T,)`` -- the ``(r, k)`` block whose rows match the
            error-correction columns -- or ``()`` at rank zero.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> res._trailing_blocks()[0].shape, VECM(y, order=2, rank=0).fit()._trailing_blocks()
            ((1, 2), ())
        """
        return (self.alpha.T,) if self.rank else ()

    @property
    def _lag_offset(self) -> int:
        """Short-run deterministic columns ahead of the lagged-difference block.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> VECM(y, order=2, rank=1, cointegration_trend="trend").fit()._lag_offset
            2
        """
        return int(self.short_run_deterministic.shape[0])

    def _coefficient_stack(self) -> npt.NDArray[np.float64]:
        """Short-run coefficients, laid out exactly as the design columns.

        Overridden because the inherited version reads :attr:`coefficients`,
        which here is the *levels* representation and was never regressed. The
        stack has to describe the regression that produced the standard errors,
        not the reparameterization of it.

        Returns:
            A ``(width, k)`` array, one row per design column: deterministic
            terms, then each lag's transposed ``Gamma``, then ``alpha.T``.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> stack = res._coefficient_stack()
            >>> stack.shape == (res.design.shape[1], 2)
            True
            >>> bool(np.allclose(stack[-1], res.alpha[:, 0]))
            True
            >>> bool(np.allclose(res.design @ stack, res.fittedvalues))
            True
        """
        blocks = [self.short_run_deterministic] if self.short_run_deterministic.shape[0] else []
        blocks += [self.gamma[i].T for i in range(self.order - 1)]
        blocks += list(self._trailing_blocks())
        return np.vstack(blocks) if blocks else np.zeros((0, self.k_endog), dtype=np.float64)

    # ------------------------------------------------------------- inference

    def stability_check(self) -> _StabilityAssessment:
        """Assess the levels companion, permitting the unit roots by design.

        A rank-``r`` system in ``k`` variables carries exactly ``k - r`` unit
        roots. Treating those as instability, which the reduced-form check does,
        would flag every correctly specified model in this family.

        Returns:
            The assessment of the levels companion matrix, with
            ``is_stable`` true when no root lies outside the unit circle and
            ``n_unit_roots`` counting the ones on it.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> check = VECM(y, order=2, rank=1).fit().stability_check()
            >>> check.is_stable, check.n_unit_roots, check.n_explosive
            (True, 1, 0)
        """
        return _StabilityAssessment.assess_stability(self.coefficients, allow_unit_roots=True)

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Point forecasts in levels.

        Iterates the levels representation from the last ``order``
        observations, extending the deterministic terms of :attr:`trend`
        past the sample, so a restricted constant folded into
        :attr:`deterministic` shifts the path exactly as it shifts the
        relations.

        Args:
            steps: Horizon.

        Returns:
            A ``(steps, k)`` array of conditional means for the levels.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> path = res.forecast(12)
            >>> path.shape, bool(np.all(np.abs(path[0] - y[-1]) < 2.0))
            ((12, 2), True)
            >>> gap = path @ res.normalized_beta()[:, 0]
            >>> bool(abs(gap[-1]) < abs(gap[0]) + 0.5)
            True
            >>> res.forecast(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        k, p = self.k_endog, self.order
        total = self.endog.shape[0]
        det = deterministic_columns(self.trend, steps, start=total + 1)
        blocks = self.coefficients
        history = [self.endog[total - i - 1] for i in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = det[h] @ self.deterministic if self.deterministic.shape[0] else np.zeros(k)
            for i in range(p):
                point = point + blocks[i] @ history[i]
            out[h] = point
            history = [point, *history[: p - 1]]
        return out

    def weak_exogeneity(self, variable: str) -> _WaldTest:
        r"""Test that a variable does not adjust to any disequilibrium.

        The null is that the variable's row of ``alpha`` is zero, so it drives
        the long-run relations without responding to them and can be treated as
        weakly exogenous for ``beta``:

        .. math::

           H_0 : \alpha_{j\cdot} = 0, \qquad W \sim \chi^2(r).

        Args:
            variable: One of :attr:`names`.

        Returns:
            A :class:`_WaldTest` with ``rank`` degrees of freedom.

        Raises:
            SpecificationError: If the variable is unknown, or the rank is zero
                and there is no adjustment to test.

        Example:
            The common trend is the first series, so it does not adjust and
            the second does:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> first, second = res.weak_exogeneity("y1"), res.weak_exogeneity("y2")
            >>> first.df, bool(first.pvalue > 0.1), bool(second.pvalue < 0.001)
            (1, True, True)
            >>> first.null
            'y1 is weakly exogenous for the cointegrating space'
            >>> VECM(y, order=2, rank=0).fit().weak_exogeneity("y1")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a rank-zero model has no cointegrating ...
        """
        if variable not in self.names:
            raise SpecificationError(
                f"unknown variable {variable!r}; expected one of {self.names}."
            )
        if not self.rank:
            raise SpecificationError(
                "a rank-zero model has no cointegrating relations, so there is nothing "
                "for a variable to be weakly exogenous with respect to."
            )
        row = self.names.index(variable)
        first = self.design.shape[1] - self.rank
        cells = [(row, first + j) for j in range(self.rank)]
        return self.coefficient_covariance.wald(
            cells, null=f"{variable} is weakly exogenous for the cointegrating space"
        )

    def granger_causality(self, cause: str, effect: str) -> _WaldTest:
        r"""Test that one variable drives another through neither channel.

        An error-correction model transmits through two routes and a test that
        checks one of them is not a test of Granger causality. The lagged
        differences carry the short-run route; the error-correction term carries
        the long-run route, where ``cause`` moves ``effect`` by shifting a
        disequilibrium that ``effect`` adjusts to. The null restricts both:

        .. math::

           H_0 : \Gamma_{i}[e, c] = 0 \;\; (i = 1, \dots, p - 1)
           \quad\text{and}\quad
           \sum_{j=1}^{r} \alpha_{ej}\, \beta_{cj} = 0 .

        The second restriction is linear in ``alpha`` once ``beta`` is treated
        as known, which is exactly the conditioning the rest of this result's
        inference already rests on, so the pair goes into one Wald statistic
        rather than two that would have to be combined by hand.

        Warning:
            Conditioning on ``beta`` makes the long-run restriction
            scale-free in it. With a single relation the restriction is
            :math:`\alpha_{e}\,\beta_{c} = 0`, and for any numerically
            non-zero :math:`\hat\beta_{c}` -- however small -- its Wald
            contribution is exactly the squared *z*-statistic of
            :math:`\alpha_{e}`. The test therefore rejects whenever
            ``effect`` adjusts to the relation at all, whatever share of it
            ``cause`` carries. Read a rejection as "no short-run channel and
            ``effect`` does not adjust" and check :math:`\beta_{c} = 0` --
            :meth:`normalized_beta` and a restricted re-estimate -- before
            calling it causality.

        Args:
            cause: The variable whose influence is restricted.
            effect: The equation the restriction applies to.

        Returns:
            A :class:`_WaldTest` with ``order - 1 + (rank > 0)`` degrees
            of freedom.

        Raises:
            SpecificationError: If either name is unknown, they are the same
                variable, or the specification has neither channel to restrict.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> res = VECM(y, order=2, rank=1).fit()
            >>> forward, back = res.granger_causality("y1", "y2"), res.granger_causality("y2", "y1")
            >>> forward.df, bool(forward.pvalue < 0.001), bool(back.pvalue > 0.1)
            (2, True, True)
            >>> forward.null
            'y1 does not Granger-cause y2'
            >>> res.granger_causality("y1", "y1")
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a variable cannot Granger-cause itself.
            >>> VECM(y, order=1, rank=0).fit().granger_causality("y1", "y2")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this specification has no lagged ...
        """
        names = self.names
        for label, value in (("cause", cause), ("effect", effect)):
            if value not in names:
                raise SpecificationError(f"{label} {value!r} is not one of {names}.")
        if cause == effect:
            raise SpecificationError("a variable cannot Granger-cause itself.")
        source, target = names.index(cause), names.index(effect)
        k, lags = self.k_endog, self.order - 1
        width = self.design.shape[1]
        if not lags and not self.rank:
            raise SpecificationError(
                "this specification has no lagged differences and no cointegrating "
                "relations, so there is no channel through which one variable could "
                "drive another."
            )
        rows: list[npt.NDArray[np.float64]] = []
        offset = self._lag_offset
        for lag in range(lags):
            restriction = np.zeros((k, width), dtype=np.float64)
            restriction[target, offset + lag * k + source] = 1.0
            rows.append(restriction.ravel())
        if self.rank:
            restriction = np.zeros((k, width), dtype=np.float64)
            first = width - self.rank
            restriction[target, first : first + self.rank] = self.beta[source]
            rows.append(restriction.ravel())
        return self.coefficient_covariance.wald_restriction(
            np.vstack(rows), null=f"{cause} does not Granger-cause {effect}"
        )

    # --------------------------------------------------------------- display

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Returns:
            ``"VECM(p, r=<rank>, <case>)"``.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> VECM(y, order=2, rank=1).fit()._comparison_label()
            'VECM(2, r=1, constant)'
        """
        return f"VECM({self.order}, r={self.rank}, {self.cointegration_trend})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        The coefficient rows are the short-run regression with its inference
        columns; below them the cointegrating vectors on the
        :meth:`normalized_beta` basis, one row per variable and per restricted
        term, with the inference columns blank because ``beta`` carries no
        standard errors here. The notes state the common-trend count, the
        conditioning on ``beta``, the identification of the space rather
        than the vector, the mismatch between the printed ``ec`` loadings and
        the printed basis, and the Cholesky ordering; an explosive root is
        flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> common = np.cumsum(rng.standard_normal(300))
            >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
            >>> table = VECM(y, order=2, rank=1).fit()._summary_table()
            >>> table.title, dict(table.metadata)["Common trends"]
            ('VECM(2) Results', '1')
            >>> [row[0] for row in table.rows[-3:]]
            ['y2: ec1', 'beta[1]: y1', 'beta[1]: y2']
            >>> table.rows[-2][1:3]
            ('1.0000', '')
        """
        criteria = self.information_criteria
        stability = self.stability_check()
        notes = [
            f"Common trends: {self.n_common_trends} of {self.k_endog}   "
            f"max |companion root| = {stability.max_modulus:.4f}",
            "Coefficients and their standard errors describe the short-run regression of "
            "differences on deterministic terms, lagged differences, and the error-"
            "correction terms; they are conditional on beta, which converges fast enough "
            "to be treated as known.",
            "Only the cointegrating space is identified without further restrictions, so "
            "read beta through normalized_beta() and treat any single vector as one basis "
            "among many.",
            "The ec rows are the loadings that pair with beta as the decomposition "
            "normalizes it, not with the basis printed below; rescaling beta rescales "
            "alpha inversely, so the product and every z-statistic are unchanged. "
            "normalized_alpha() returns the loadings on the printed basis.",
            _CHOLESKY_NOTE,
        ]
        if stability.max_modulus > 1.0 + 1e-8:
            notes.insert(0, _UNSTABLE_NOTE)
        rows = self._coefficient_rows()
        if self.rank:
            beta_rows = self.normalized_beta()
            labels = (
                *self.names,
                "const" if self.cointegration_trend == "restricted_constant" else "trend",
            )[: beta_rows.shape[0]]
            rows = rows + tuple(
                (f"beta[{j + 1}]: {label}", f"{beta_rows[i, j]:.4f}", "", "", "", "", "")
                for j in range(self.rank)
                for i, label in enumerate(labels)
            )
        return SummaryTable(
            title=f"VECM({self.order}) Results",
            metadata=(
                ("Model", f"VECM({self.order}, r={self.rank})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Rank", f"{self.rank}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Deterministic", self.cointegration_trend),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("Common trends", f"{self.n_common_trends}"),
            ),
            columns=self._coefficient_columns(),
            rows=rows,
            notes=tuple(notes),
        )


class VECM(_VectorErrorCorrectionModel[VECMResult]):
    r"""Vector error-correction model, estimated by Johansen's reduced-rank ML.

    The closed specification: every variable in the panel gets an equation,
    the rank is fixed by the caller, and the Johansen case names where a
    constant or trend sits relative to the cointegrating space. ``order``
    counts lags of the *levels* system, so ``order=p`` carries ``p - 1``
    lagged differences and the inherited ``lag_order_selection`` chooses it
    the standard way, on the unrestricted levels VAR. The rank is a property
    of the data, not of this object, so ``rank_test()`` -- the trace and
    maximum-eigenvalue sequence with simulated critical values -- can be read
    from a model built at any rank before committing to one.

    Attributes:
        _endog: The validated ``(nobs, k)`` panel of levels.
        _names: Variable labels.
        _order: Lags of the levels system.
        _trend: The unrestricted deterministic terms the short-run regression
            sees, mapped from the Johansen case.
        _prior: The base slot for a shrinkage prior, unused by this family.
        _rank: The cointegrating rank.
        _trend_case: The Johansen case as given.
        _exog: The exogenous block, ``(nobs, 0)`` for this closed model.
        _exog_names: Exogenous labels, empty here.
        _contemporaneous: Whether a current exogenous difference would enter;
            irrelevant without a block.

    See Also:
        * :class:`VECMResult` -- the record ``fit()`` returns.
        * :class:`VECMX` -- the conditional model with weakly exogenous
          integrated regressors.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the levels system this reparameterizes; fit it when the rank
          test selects ``k``.

    References:
        Johansen, S. (1991). Estimation and hypothesis testing of
        cointegration vectors in Gaussian vector autoregressive models.
        *Econometrica*, 59(6), 1551-1580.

        Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated
        Vector Autoregressive Models*. Oxford University Press.

    Example:
        Choose the rank from the data, then fit at it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> common = np.cumsum(rng.standard_normal(300))
        >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
        >>> model = VECM(y, order=2, rank=1)
        >>> model.rank_test(simulations=2000).selected_rank()
        1
        >>> res = model.fit()
        >>> res.n_common_trends, res.normalized_beta().ravel().round(2)
        (1, array([ 1.  , -0.49]))
    """

    __slots__ = ()

    def fit(self) -> VECMResult:
        """Estimate the system at the specified rank and return the result.

        Returns:
            A :class:`VECMResult`.

        Example:
            >>> import numpy as np
            >>> y = np.cumsum(np.random.default_rng(0).standard_normal((200, 2)), axis=0)
            >>> res = VECM(y, order=2, rank=1, cointegration_trend="restricted_constant").fit()
            >>> res.rank, res.trend, res.short_run_deterministic.shape, res.deterministic.shape
            (1, 'c', (0, 2), (1, 2))
        """
        return VECMResult._from_fit(self._fit_family(), self)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VECMXResult(VECMResult):
    r"""A fitted conditional vector error-correction model.

    The Pesaran-Shin-Smith specification: ``k_y`` modelled variables share a
    cointegrating space with ``k_x`` weakly exogenous integrated regressors,
    and only the modelled equations are estimated,

    .. math::

       \Delta y_t = \alpha \beta' z_{t-1} + \sum_{i=1}^{p-1} \Gamma_i\, \Delta z_{t-i}
       + \Lambda_0\, \Delta x_t + \mu_t + u_t, \qquad z_t = (y_t', x_t')',

    so ``beta`` spans the combined vector, ``Gamma`` is ``(k_y, k_y + k_x)``
    because lagged differences of both blocks enter, and ``impact``
    (:math:`\Lambda_0`) carries the contemporaneous response to the exogenous
    differences. The regressors are *weakly exogenous*: they may share the
    long-run relations, but their own equations are not written and their
    adjustment to the disequilibria is assumed away rather than estimated.

    Everything that reads the estimated regression is inherited and correct:
    ``params``, ``bse``, ``pvalues``, ``conf_int``, the residual diagnostics,
    the weak-exogeneity test, and the two-channel Granger test. What is *not*
    inherited is the half of the surface that needs a closed system. Impulse
    responses, the variance decomposition, the historical decomposition, the
    companion matrix, and the stability check all require a law of motion for
    every integrated variable in the system, and this model deliberately does
    not have one for ``x`` -- that omission is the specification, not a gap.
    Those six raise rather than returning a number computed from the modelled
    block alone, which would look like an impulse response and be an artifact.

    :meth:`forecast` survives the same test and is kept: it needs a *path* for
    the exogenous block, not a model of it, so the caller supplies one exactly
    as for a VARX. :meth:`to_varx` is the other thing a conditional model
    does have -- its own equations in levels -- and is what
    :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR` consumes
    to close several such units into one system.

    Note:
        The inherited levels fields are placeholders here: ``coefficients``
        is ``(0, k_y, k_y)`` and ``deterministic`` is ``(0, k_y)``, because
        the levels representation of a conditional model is not square and
        lives in :meth:`to_varx` instead. ``n_common_trends`` counts the
        joint system, ``k_y + k_x - r``, since the exogenous block carries
        its own unit roots. The rank may run to ``k_y`` rather than
        ``k_y - 1``: every modelled variable cointegrating with the exogenous
        block is a valid conditional specification.

    Attributes:
        exog: The weakly exogenous sample.
        exog_names: Exogenous labels.
        impact: ``(k_y, k_x)`` contemporaneous response to exogenous
            differences, zero-width when the model excludes it.
        contemporaneous: Whether the current exogenous difference is included.

    See Also:
        * :class:`VECMX` -- the model whose ``fit()`` returns this record.
        * :class:`VECMResult` -- the closed parent, whose short-run inference
          this result inherits and whose propagation surface it refuses.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARXResult`
          -- the conditional levels model with the same refusals and the
          same ``to_varx()`` currency.
        * :class:`~cultivars.multivariate.reduced_form.closed_global.GVARResult`
          -- the closed system assembled from units like this one, where the
          refused members become available.

    References:
        Pesaran, M. H., Shin, Y., & Smith, R. J. (2000). Structural analysis
        of vector error correction models with exogenous I(1) variables.
        *Journal of Econometrics*, 97(2), 293-343.

        Harbo, I., Johansen, S., Nielsen, B., & Rahbek, A. (1998). Asymptotic
        inference on cointegrating rank in partial systems. *Journal of
        Business & Economic Statistics*, 16(4), 388-399.

    Example:
        One modelled variable error-corrects toward an exogenous random walk
        while a second modelled variable wanders on its own. The conditional
        fit recovers the relation :math:`y_1 - x_1`, finds that ``y1`` adjusts
        and ``y2`` does not, and refuses an impulse response while forecasting
        along a supplied path for ``x``:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = np.cumsum(rng.standard_normal(300))
        >>> y1 = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y1[t] = y1[t - 1] + 0.4 * (x[t - 1] - y1[t - 1]) + 0.5 * rng.standard_normal()
        >>> y = np.column_stack([y1, np.cumsum(rng.standard_normal(300))])
        >>> res = VECMX(y, x[:, None], order=2, rank=1).fit()
        >>> res.k_exog, res.n_common_trends, res.beta.shape
        (1, 2, (3, 1))
        >>> res.normalized_beta().ravel().round(1)
        array([ 1.,  0., -1.])
        >>> adjusts = [res.weak_exogeneity(name).pvalue < 0.05 for name in res.names]
        >>> adjusts
        [True, False]
        >>> res.irf(4)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: a conditional vector error-correction model has ...
        >>> path = res.forecast(6, exog_future=np.full((6, 1), x[-1]))
        >>> gap = path[:, 0] - x[-1]
        >>> path.shape, bool(abs(gap[-1]) < abs(gap[0]))
        ((6, 2), True)
    """

    exog: npt.NDArray[np.float64]
    """The ``(nobs, k_x)`` weakly exogenous integrated regressors, in levels."""

    exog_names: tuple[str, ...]
    """Labels of the exogenous block, ``x1 ... xk`` unless given."""

    impact: npt.NDArray[np.float64]
    r"""The ``(k_y, k_x)`` coefficients :math:`\Lambda_0` on the current exogenous differences.

    Zero-width when :attr:`contemporaneous` is false.
    """

    contemporaneous: bool
    """Whether the current exogenous difference entered the short-run regression."""

    @classmethod
    def _from_fit(
        cls, fit: _VectorErrorCorrectionFit, model: _VectorErrorCorrectionModel[Self]
    ) -> Self:
        """Assemble the public result from the internal fit and its model.

        Annotated at the parent's binding rather than narrowed to the
        conditional model, which would be a contravariance violation. Nothing
        is lost: the exogenous block lives on the shared base, because a closed
        system is the case where it happens to be empty.

        Args:
            fit: The packed reduced-rank fit, with its ``impact`` block.
            model: The model that produced it, read for both samples, both
                label sets, the order, the rank, the Johansen case and the
                contemporaneous flag.

        Returns:
            A populated conditional result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> y = x + rng.standard_normal((200, 1))
            >>> model = VECMX(y, x, order=1, rank=1, contemporaneous=False)
            >>> res = VECMXResult._from_fit(model._fit_family(), model)
            >>> res.contemporaneous, res.impact.shape, res.exog_names
            (False, (1, 0), ('x1',))
        """
        case = cast(CointegrationTrend, model.cointegration_trend)
        return cls(
            endog=model.endog,
            exog=model.exog,
            names=model.names,
            exog_names=model.exog_names,
            order=model.order,
            rank=model.rank,
            cointegration_trend=case,
            trend=_LEVELS_TREND[case],
            alpha=fit.alpha,
            beta=fit.beta,
            gamma=fit.gamma,
            impact=fit.impact,
            contemporaneous=model.contemporaneous,
            coefficients=fit.coefficients,
            deterministic=fit.deterministic,
            short_run_deterministic=fit.short_run_deterministic,
            eigenvalues=fit.eigenvalues,
            sigma_u=fit.sigma_u,
            sigma_ml=fit.sigma_ml,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            design=fit.design,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
        )

    @property
    def k_exog(self) -> int:
        """Weakly exogenous integrated regressors.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 2)), axis=0)
            >>> y = x[:, :1] + rng.standard_normal((200, 1))
            >>> VECMX(y, x, order=1, rank=1).fit().k_exog
            2
        """
        return len(self.exog_names)

    @property
    def n_common_trends(self) -> int:
        """Unit roots the joint system carries, ``k_y + k_x - r``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 2)), axis=0)
            >>> y = x[:, :1] + rng.standard_normal((200, 1))
            >>> VECMX(y, x, order=1, rank=1).fit().n_common_trends
            2
        """
        return self.k_endog + self.k_exog - self.rank

    def _refuse(self, what: str) -> None:
        """Raise the shared explanation for a quantity the conditional model lacks.

        Args:
            what: The quantity being asked for, as a noun phrase that fits
                "``{what}`` is not defined for it".

        Raises:
            SpecificationError: Always, with ``_CONDITIONAL_REFUSAL`` filled
                in with the quantity and the exogenous labels.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res._refuse("a thing")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a conditional vector error-correction model ...
        """
        raise SpecificationError(_CONDITIONAL_REFUSAL.format(what=what, names=self.exog_names))

    def _lag_labels(self) -> tuple[str, ...]:
        """Lagged-difference column names over the joint vector.

        Returns:
            ``"D.<name>.L<lag>"`` over the modelled then the exogenous
            labels, lag-major, matching the ``(k_y, k_y + k_x)`` layout of
            each ``Gamma``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=2, rank=1).fit()
            >>> res._lag_labels()
            ('D.y1.L1', 'D.x1.L1')
        """
        joint = (*self.names, *self.exog_names)
        return tuple(f"D.{source}.L{lag + 1}" for lag in range(self.order - 1) for source in joint)

    def _trailing_labels(self) -> tuple[str, ...]:
        """Contemporaneous exogenous differences, then the error-correction terms.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> y = x + rng.standard_normal((200, 1))
            >>> VECMX(y, x, order=1, rank=1).fit()._trailing_labels()
            ('D.x1', 'ec1')
            >>> VECMX(y, x, order=1, rank=1, contemporaneous=False).fit()._trailing_labels()
            ('ec1',)
        """
        head = tuple(f"D.{name}" for name in self.exog_names) if self.contemporaneous else ()
        return head + tuple(f"ec{i + 1}" for i in range(self.rank))

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The impact block, then the adjustment loadings, in design order.

        Returns:
            ``(impact.T, alpha.T)`` with either omitted when absent, each
            laid out with one row per design column.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> [block.shape for block in res._trailing_blocks()]
            [(1, 1), (1, 1)]
        """
        head = (self.impact.T,) if self.contemporaneous else ()
        return head + ((self.alpha.T,) if self.rank else ())

    def _coefficient_stack(self) -> npt.NDArray[np.float64]:
        """Short-run coefficients, laid out exactly as the design columns.

        Returns:
            A ``(width, k_y)`` array: deterministic terms, each lag's
            transposed ``Gamma`` over the joint vector, the transposed impact
            block, then ``alpha.T``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=2, rank=1).fit()
            >>> stack = res._coefficient_stack()
            >>> stack.shape, bool(np.allclose(res.design @ stack, res.fittedvalues))
            ((5, 1), True)
        """
        blocks = [self.short_run_deterministic] if self.short_run_deterministic.shape[0] else []
        blocks += [self.gamma[i].T for i in range(self.order - 1)]
        blocks += list(self._trailing_blocks())
        return np.vstack(blocks) if blocks else np.zeros((0, self.k_endog), dtype=np.float64)

    def to_var(self) -> npt.NDArray[np.float64]:
        """Unavailable: a conditional model has no levels representation.

        Raises:
            SpecificationError: Always; see :meth:`to_varx` for the
                conditional levels form that does exist.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.to_var()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so a levels vector autoregression is ...
        """
        self._refuse("a levels vector autoregression")
        raise AssertionError  # pragma: no cover

    def to_varx(self) -> _ConditionalLevels:
        r"""The conditional levels form this error-correction model implies.

        Not :meth:`to_var`, which stays unavailable: that would be a closed
        system and this specification has none. What *is* recoverable is the
        unit's own equations in levels -- how its variables respond to their own
        past and to the foreign block -- and that is exactly what a global
        system consumes when it links units together. The distinction is the
        difference between "no law of motion for x" and "no equations at all".

        Splitting :math:`\Pi = \alpha\beta'` and each :math:`\Gamma_i` into
        their own and foreign columns and undoing the differencing gives

        .. math::

           \Phi_1 = I + \Pi_y + \Gamma_{1,y}, \quad
           \Phi_i = \Gamma_{i,y} - \Gamma_{i-1,y}, \quad
           \Phi_p = -\Gamma_{p-1,y},

        and the same recursion for the foreign lags with
        :math:`\Lambda_1 = \Pi_x - \Lambda_0 + \Gamma_{1,x}`, so that
        :math:`y_t = \sum_i \Phi_i y_{t-i} + \Lambda_0 x_t + \sum_i \Lambda_i
        x_{t-i} + \mu_t + u_t` reproduces every fitted value exactly.

        Returns:
            A :class:`_ConditionalLevels` record with ``phi`` of shape
            ``(p, k_y, k_y)``, ``impact`` of ``(k_y, k_x)`` and ``exog_lags``
            of ``(p, k_y, k_x)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal(300))
            >>> y = (x + rng.standard_normal(300))[:, None]
            >>> res = VECMX(y, x[:, None], order=2, rank=1).fit()
            >>> levels = res.to_varx()
            >>> levels.phi.shape, levels.impact.shape, levels.exog_lags.shape
            ((2, 1, 1), (1, 1), (2, 1, 1))
            >>> own = y[1:-1] @ levels.phi[0].T + y[:-2] @ levels.phi[1].T
            >>> foreign = levels.impact[0, 0] * x[2:] + levels.exog_lags[0, 0, 0] * x[1:-1]
            >>> foreign = foreign + levels.exog_lags[1, 0, 0] * x[:-2]
            >>> fitted = own[:, 0] + foreign + levels.deterministic[0, 0]
            >>> bool(np.allclose(y[2:, 0], fitted + res.resid[:, 0]))
            True
        """
        k, m, p = self.k_endog, self.k_exog, self.order
        joint = self.alpha @ self.beta[: k + m].T
        own_long, foreign_long = joint[:, :k], joint[:, k:]
        own_short = [self.gamma[lag][:, :k] for lag in range(p - 1)]
        foreign_short = [self.gamma[lag][:, k:] for lag in range(p - 1)]
        phi = np.zeros((p, k, k), dtype=np.float64)
        exog_lags = np.zeros((p, k, m), dtype=np.float64)
        phi[0] = np.eye(k) + own_long
        exog_lags[0] = foreign_long - self.impact
        if p > 1:
            phi[0] = phi[0] + own_short[0]
            exog_lags[0] = exog_lags[0] + foreign_short[0]
            for lag in range(1, p - 1):
                phi[lag] = own_short[lag] - own_short[lag - 1]
                exog_lags[lag] = foreign_short[lag] - foreign_short[lag - 1]
            phi[p - 1] = -own_short[p - 2]
            exog_lags[p - 1] = -foreign_short[p - 2]
        return _ConditionalLevels(
            phi=phi,
            impact=self.impact,
            exog_lags=exog_lags,
            deterministic=self.short_run_deterministic,
            names=self.names,
            exog_names=self.exog_names,
        )

    @property
    def companion(self) -> npt.NDArray[np.float64]:
        """Unavailable: a conditional model has no closed companion.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.companion  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so the companion matrix is not defined ...
        """
        self._refuse("the companion matrix")
        raise AssertionError  # pragma: no cover

    def stability_check(self) -> _StabilityAssessment:
        """Unavailable: stability is a property of the closed system.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.stability_check()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so a stability check is not defined ...
        """
        self._refuse("a stability check")
        raise AssertionError  # pragma: no cover

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs a law of motion for the exogenous block.

        Args:
            horizon: Ignored.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.ma_representation(4)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so a moving-average representation is ...
        """
        self._refuse("a moving-average representation")
        raise AssertionError  # pragma: no cover

    def irf(
        self, horizon: int = 20, *, orthogonalized: bool = True, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs a law of motion for the exogenous block.

        Args:
            horizon: Ignored.
            orthogonalized: Ignored.
            cumulative: Ignored.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.irf(4)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so an impulse response is not defined ...
        """
        self._refuse("an impulse response")
        raise AssertionError  # pragma: no cover

    def fevd(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs a law of motion for the exogenous block.

        Args:
            horizon: Ignored.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.fevd(4)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so a variance decomposition is not ...
        """
        self._refuse("a variance decomposition")
        raise AssertionError  # pragma: no cover

    def historical_decomposition(self) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs a law of motion for the exogenous block.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.historical_decomposition()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: ... so a historical decomposition is not ...
        """
        self._refuse("a historical decomposition")
        raise AssertionError  # pragma: no cover

    def forecast(
        self, steps: int = 1, *, exog_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        r"""Point forecasts conditional on a future path for the exogenous block.

        Iterates the short-run equation forward in its own coordinates:
        at each step the disequilibrium :math:`\hat\beta' z_{t-1}` is
        rebuilt from the joint path (with the restricted constant or trend
        appended when the Johansen case carries one), the lagged differences
        are read off the path, the contemporaneous impact uses the supplied
        exogenous change, and the modelled levels advance by the predicted
        difference.

        Args:
            steps: Horizon.
            exog_future: A ``(steps, k_x)`` path in the column order of
                :attr:`exog_names`. Required.

        Returns:
            A ``(steps, k_y)`` array of conditional means for the modelled
            levels.

        Raises:
            SpecificationError: If ``steps`` is not positive or the path is
                omitted.
            DimensionError: If the path has the wrong shape.

        Example:
            Holding ``x`` at its last value, the modelled variable closes its
            gap to it over the horizon:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal(300))
            >>> y1 = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y1[t] = y1[t - 1] + 0.4 * (x[t - 1] - y1[t - 1]) + 0.5 * rng.standard_normal()
            >>> res = VECMX(y1[:, None], x[:, None], order=2, rank=1).fit()
            >>> path = res.forecast(8, exog_future=np.full((8, 1), x[-1]))
            >>> gap = np.abs(path[:, 0] - x[-1])
            >>> path.shape, bool(gap[-1] < 0.5 * gap[0])
            ((8, 1), True)
            >>> res.forecast(3)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a conditional forecast needs the exogenous ...
            >>> res.forecast(3, exog_future=np.zeros((3, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: exog_future has 2 columns but the model was ...
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        if exog_future is None:
            raise SpecificationError(
                "a conditional forecast needs the exogenous path, so it cannot be produced "
                f"from the fitted model alone: pass exog_future with {steps} rows and "
                f"{self.k_exog} columns for {self.exog_names}. This model holds no process "
                "for x and will not invent one."
            )
        future = validate_exog_matrix(exog_future, nobs=steps, label="exog_future")
        if future.shape[1] != self.k_exog:
            raise DimensionError(
                f"exog_future has {future.shape[1]} columns but the model was fitted with "
                f"{self.k_exog}."
            )
        k, p = self.k_endog, self.order
        total = self.endog.shape[0]
        det = deterministic_columns(
            _UNRESTRICTED_TREND[cast(CointegrationTrend, self.cointegration_trend)],
            steps,
            start=total + 1,
        )
        joint = np.column_stack([self.endog, self.exog])
        path = np.vstack([joint, np.column_stack([np.zeros((steps, k)), future])])
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            row = total + h
            level = path[row - 1]
            point = (
                det[h] @ self.short_run_deterministic
                if self.short_run_deterministic.shape[0]
                else np.zeros(k)
            )
            if self.rank:
                extended = level
                if self.cointegration_trend == "restricted_constant":
                    extended = np.append(level, 1.0)
                elif self.cointegration_trend == "restricted_trend":
                    extended = np.append(level, float(row))
                point = point + self.alpha @ (self.beta.T @ extended)
            for lag in range(p - 1):
                point = point + self.gamma[lag] @ (path[row - lag - 1] - path[row - lag - 2])
            if self.contemporaneous:
                point = point + self.impact @ (path[row, k:] - path[row - 1, k:])
            out[h] = level[:k] + point
            path[row, :k] = out[h]
        return out

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Returns:
            ``"VECMX(p, r=<rank>, kx=<k_x>)"``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=2, rank=1).fit()
            >>> res._comparison_label()
            'VECMX(2, r=1, kx=1)'
        """
        return f"VECMX({self.order}, r={self.rank}, kx={self.k_exog})"


class VECMX(_ExogenousVectorErrorCorrectionModel[VECMXResult]):
    r"""Conditional vector error-correction model with weakly exogenous I(1) regressors.

    The partial system of Pesaran, Shin and Smith: the modelled block ``y``
    gets equations, the exogenous block ``x`` is carried through the
    cointegrating space, the lagged differences and, by default, a
    contemporaneous difference, but is never modelled. The rank may reach
    ``k_y``, and the inherited ``rank_test()`` reads it against the
    conditional null distribution, whose critical values are materially
    larger than the closed-system ones. This is the unit specification a
    :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR` links,
    and the only unit kind whose weak-exogeneity assumption that assembler
    can test.

    Attributes:
        _endog: The validated ``(nobs, k_y)`` modelled panel.
        _names: Modelled variable labels.
        _order: Lags of the levels system.
        _trend: The unrestricted deterministic terms, mapped from the Johansen
            case.
        _prior: The base slot for a shrinkage prior, unused by this family.
        _rank: The cointegrating rank.
        _trend_case: The Johansen case as given.
        _exog: The validated ``(nobs, k_x)`` exogenous block.
        _exog_names: Exogenous labels.
        _contemporaneous: Whether the current exogenous difference enters.

    See Also:
        * :class:`VECMXResult` -- the record ``fit()`` returns.
        * :class:`VECM` -- the closed model, the case where the exogenous
          block is empty.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
          -- the conditional model in levels.

    References:
        Pesaran, M. H., Shin, Y., & Smith, R. J. (2000). Structural analysis
        of vector error correction models with exogenous I(1) variables.
        *Journal of Econometrics*, 97(2), 293-343.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = np.cumsum(rng.standard_normal((300, 1)), axis=0)
        >>> other = np.cumsum(rng.standard_normal(300))
        >>> y = np.column_stack([x[:, 0] + rng.standard_normal(300), other])
        >>> model = VECMX(y, x, order=2, rank=1)
        >>> model.rank_test(simulations=2000).selected_rank()
        1
        >>> res = model.fit()
        >>> res.k_exog, res.rank, (res.normalized_beta().ravel().round(1) + 0.0)
        (1, 1, array([ 1.,  0., -1.]))
    """

    __slots__ = ()

    def fit(self) -> VECMXResult:
        """Estimate the conditional system and return the fitted result.

        Returns:
            A :class:`VECMXResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
            >>> res = VECMX(x + rng.standard_normal((200, 1)), x, order=1, rank=1).fit()
            >>> res.k_endog, res.k_exog, res.design.shape[1]
            (1, 1, 3)
        """
        return VECMXResult._from_fit(self._fit_family(), self)
