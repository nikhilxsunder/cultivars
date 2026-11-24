# filepath: /src/cultivars/summary.py
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
r"""The reporting surface: the records every fitted result renders itself through.

A fitted result is a frozen record, and ``summary()`` does not format it; it
returns a :class:`SummaryTable`, a second record holding the structure of
the report -- a title, a block of scalar metadata, one coefficient table,
and the closing notes that state what the numbers assume -- which renders
on demand as fixed-width text, as HTML in a notebook, or as a dataframe.
:class:`InformationCriteria` is the record behind every
``information_criteria`` property: the three penalized likelihoods,

.. math::

   \mathrm{AIC} = -2\ell + 2k, \qquad
   \mathrm{BIC} = -2\ell + k \log n, \qquad
   \mathrm{HQIC} = -2\ell + 2k \log\log n,

for a maximized log-likelihood :math:`\ell`, parameter count :math:`k` and
sample size :math:`n`, with the convention for :math:`k` stated by the
result that produced it.

The two records are defined in ``cultivars_engine.summary`` and this module
is their public name. They are the same objects, not subclasses, so every
``summary()`` and ``information_criteria`` in the package returns an
instance of the classes imported from here.

Example:
    >>> import numpy as np
    >>> from cultivars.summary import InformationCriteria, SummaryTable
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> res = ARIMA(np.random.default_rng(0).standard_normal(120), order=(1, 0, 0)).fit()
    >>> isinstance(res.summary(), SummaryTable)
    True
    >>> isinstance(res.information_criteria, InformationCriteria)
    True
    >>> ic = InformationCriteria.from_likelihood(-100.0, 50, 3)
    >>> round(ic.aic, 1), bool(ic.aic < ic.hqic < ic.bic)
    (206.0, True)
"""

from __future__ import annotations

from cultivars_engine.summary import InformationCriteria, SummaryTable

__all__ = ["InformationCriteria", "SummaryTable"]
