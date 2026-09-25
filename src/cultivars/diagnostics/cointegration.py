# filepath: /src/cultivars/diagnostics/cointegration.py
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
r"""Cointegration rank: the Johansen trace and maximum-eigenvalue sequences.

A system of :math:`k` integrated series has cointegrating rank
:math:`r` when :math:`r` independent linear combinations of it are
stationary, and the rank is the first thing a vector error-correction
model must know. Johansen's (1991) reduced-rank regression estimates it
from the squared canonical correlations
:math:`\hat\lambda_1 \ge \cdots \ge \hat\lambda_k` between the
differences and the lagged levels, each corrected for the short-run
dynamics, and tests it with two sequences read down from :math:`r = 0`,

.. math::

   \lambda_{\text{trace}}(r) = -T \sum_{i = r + 1}^{k} \ln(1 - \hat\lambda_i),
   \qquad
   \lambda_{\max}(r) = -T \ln(1 - \hat\lambda_{r + 1}),

the first against the alternative of full rank and the second against
rank :math:`r + 1`. The conventional reading stops at the first
:math:`r` not rejected. Neither statistic has a standard limit: both are
functionals of a :math:`(k - r)`-dimensional Brownian motion whose form
depends on where the deterministic terms sit relative to the
cointegrating space, the five cases of Johansen (1995), and, when
weakly exogenous integrated regressors are conditioned on, on the
number of them (Pesaran, Shin & Smith 2000). The record here carries
the eigenvalues, both sequences, and empirical p-values from that limit
simulated for the case and the exogenous count in hand.

Two commitments shape the surface. First, the test is not a function in
this module. The eigenvalue problem belongs to the model that solves
it, so :meth:`~cultivars.multivariate.reduced_form.error_correction.VECM.rank_test`
produces the record, independent of the rank the model was constructed
with; this module is where the record is *documented*, its public home
under the name the rest of the package refers to. Second, the p-values
are simulated, not tabulated, and say so: the record carries the number
of replications, so the smallest resolvable p-value is visible, and the
table names the deterministic case the null was drawn under, because a
conditional statistic read against the unconditional table would
over-reject.

Layout. :class:`JohansenRankTest` is the frozen record, with
:meth:`~JohansenRankTest.selected_rank` for the sequential reading under
either statistic and :meth:`~JohansenRankTest.to_table` for the summary.
It is defined in ``_internals`` as the return type of the
error-correction models' ``rank_test`` and re-exported here. The
statistics are computed in ``_internals`` from the model's
cointegration moments; the null is simulated by ``_core``'s
``simulate_cointegration_null``, one Brownian-functional draw per
replication for each ``k - r``.

References:
    Johansen, S. (1991). Estimation and hypothesis testing of
    cointegration vectors in Gaussian vector autoregressive models.
    *Econometrica*, 59(6), 1551-1580.

    Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated
    Vector Autoregressive Models*. Oxford University Press.

    Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic
    distribution of the maximum likelihood cointegration rank test
    statistics. *Oxford Bulletin of Economics and Statistics*, 54(3),
    461-472.

    MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical
    distribution functions of likelihood ratio tests for cointegration.
    *Journal of Applied Econometrics*, 14(5), 563-577.

    Pesaran, M. H., Shin, Y., & Smith, R. J. (2000). Structural analysis
    of vector error correction models with exogenous I(1) variables.
    *Journal of Econometrics*, 97(2), 293-343.

    Reinsel, G. C., & Ahn, S. K. (1992). Vector autoregressive models
    with unit roots and reduced rank structure: Estimation, likelihood
    ratio test, and forecasting. *Journal of Time Series Analysis*,
    13(4), 353-375.

Example:
    Three integrated series of which two share a common trend, so the
    rank is one; both sequences find it:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.error_correction import VECM
    >>> rng = np.random.default_rng(0)
    >>> common = np.cumsum(rng.standard_normal(300))
    >>> y = np.column_stack([
    ...     common + rng.standard_normal(300),
    ...     0.5 * common + rng.standard_normal(300),
    ...     np.cumsum(rng.standard_normal(300)),
    ... ])
    >>> test = VECM(y, order=1, rank=1).rank_test(simulations=2000)
    >>> test
    JohansenRankTest(k=3, nobs=299, deterministic='constant', trace_rank=1, max_eigenvalue_rank=1)
    >>> test.trace_pvalue.round(2)
    array([0.  , 0.21, 0.47])
    >>> isinstance(test, JohansenRankTest)
    True
"""

from __future__ import annotations

from .._internals import _JohansenRankTest as JohansenRankTest

__all__ = ["JohansenRankTest"]
