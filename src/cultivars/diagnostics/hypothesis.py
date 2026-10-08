# filepath: /src/cultivars/diagnostics/hypothesis.py
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
r"""Hypothesis-test records: one statistic, one p-value, one verdict, one table.

Every test in the package returns a record rather than a tuple, and
every record descends from :class:`HypothesisTest`: a statistic, a
p-value that may be ``None``, :meth:`~HypothesisTest.reject` at a level,
and a summary rendered once as text, as notebook HTML, and as a
:class:`~cultivars._core.SummaryTable`. A study that collects verdicts
from a unit-root test, a Granger-causality test, and a structural-break
test then holds one type, not a union of three. Two branches specialize
the base by how the null distribution is known. A
:class:`ChiSquaredTest` refers a statistic to :math:`\chi^2_{df}` and
carries the degrees of freedom; its two members are the
:class:`LikelihoodRatioTest`,

.. math::

   LR = 2\,(\ell_{\text{unrestricted}} - \ell_{\text{restricted}})
   \;\xrightarrow{d}\; \chi^2_{df},

named by its construction from two nested fits on a common sample, and
the :class:`WaldTest`,

.. math::

   W = (R\hat\theta - r)^\top
       \bigl(R\,\widehat{\operatorname{Var}}(\hat\theta)\,R^\top\bigr)^{-1}
       (R\hat\theta - r)
   \;\xrightarrow{d}\; \chi^2_{\operatorname{rank} R},

which is a form rather than a hypothesis, the same quadratic answering
Granger causality, residual autocorrelation, normality, and conditional
heteroskedasticity, and which therefore carries its ``null`` as a
sentence. A :class:`TabulatedTest` is for the non-standard limits of the
unit-root and structural-break literature: it carries critical values at
the 1%, 5%, and 10% levels, a ``lower_tail`` flag saying which way
rejection lies, and a p-value only when a response surface or a
simulated limit supplies one, reading the table otherwise; several
statistics of the same test on the same sample travel as ``companions``,
each a full record.

Two commitments shape the surface. First, a record that cannot answer
refuses rather than guesses: :meth:`~HypothesisTest.reject` on a
tabulated test without a p-value raises unless ``alpha`` is a tabulated
level, and never interpolates between the three columns. Second, the
verdict is at the record's own level and says so: the repr names the
default level, the table's header carries ``Verdict at 5%``, and a Wald
repr reads as a sentence with its null, because four Wald verdicts on
one model would otherwise be indistinguishable in a log.

Layout. The five names here are the public homes of records defined in
``_internals``, where the models that produce them live:
:class:`HypothesisTest` and :class:`TabulatedTest` are the bases the
diagnostics modules subclass, :class:`~cultivars.diagnostics.unit_roots.UnitRootTest`
and :class:`~cultivars.diagnostics.breaks.BreakTest` among them;
:class:`ChiSquaredTest`, :class:`LikelihoodRatioTest`, and
:class:`WaldTest` are returned by the fitted models' diagnostic methods,
``likelihood_ratio_test``, ``granger_causality``, ``portmanteau_test``,
``normality_test``, ``arch_test``, and the covariance objects' ``wald``
and ``wald_restriction``. Nothing in this module computes a statistic;
it names the records so that annotations, ``isinstance`` checks, and
the documentation refer to public types.

References:
    Wald, A. (1943). Tests of statistical hypotheses concerning several
    parameters when the number of observations is large. *Transactions
    of the American Mathematical Society*, 54(3), 426-482.

    Wilks, S. S. (1938). The large-sample distribution of the likelihood
    ratio for testing composite hypotheses. *Annals of Mathematical
    Statistics*, 9(1), 60-62.

    Engle, R. F. (1984). Wald, likelihood ratio, and Lagrange multiplier
    tests in econometrics. In *Handbook of Econometrics, Volume 2*
    (pp. 775-826). North-Holland.

    Gregory, A. W., & Veall, M. R. (1985). Formulating Wald tests of
    nonlinear restrictions. *Econometrica*, 53(6), 1465-1468.

Example:
    A Granger-causality test is a :class:`WaldTest`, a nested-model
    comparison a :class:`LikelihoodRatioTest`, and both are
    :class:`HypothesisTest` records with the same ``reject``:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((200, 2))
    >>> for t in range(1, 200):
    ...     y[t] = np.array([[0.5, 0.3], [0.0, 0.5]]) @ y[t - 1] + rng.standard_normal(2)
    >>> res = VAR(y, order=1, names=["a", "b"]).fit()
    >>> res.granger_causality("b", "a")
    WaldTest(statistic=13.4453, df=1, pvalue=0.0002456, reject at 5%: 'b does not Granger-cause a')
    >>> wider = VAR(y, order=1, trend="ct", names=["a", "b"]).fit()
    >>> test = res.likelihood_ratio_test(wider)
    >>> test
    LikelihoodRatioTest(statistic=2.7665, df=2, pvalue=0.2508)
    >>> isinstance(test, HypothesisTest), isinstance(test, ChiSquaredTest), test.reject()
    (True, True, False)
"""

from __future__ import annotations

from .._internals import _ChiSquaredTest as ChiSquaredTest
from .._internals import _HypothesisTest as HypothesisTest
from .._internals import _LikelihoodRatioTest as LikelihoodRatioTest
from .._internals import _TabulatedTest as TabulatedTest
from .._internals import _WaldTest as WaldTest

__all__ = ["ChiSquaredTest", "HypothesisTest", "LikelihoodRatioTest", "TabulatedTest", "WaldTest"]
