# filepath: /src/cultivars/__about__.py
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

A fitted result is a frozen record, and ``summary()`` does not format
it; it returns a :class:`SummaryTable`, a second record holding the
*structure* of the report -- a title, a block of scalar metadata, one
coefficient table, and the closing notes that state what the numbers
assume -- which renders on demand as fixed-width text, as HTML in a
notebook, or as a dataframe. :class:`InformationCriteria` is the record
behind every ``information_criteria`` property: the three penalized
likelihoods,

.. math::

   \mathrm{AIC} = -2\ell + 2k, \qquad
   \mathrm{BIC} = -2\ell + k \log n, \qquad
   \mathrm{HQIC} = -2\ell + 2k \log\log n,

for a maximized log-likelihood :math:`\ell`, parameter count :math:`k`
and sample size :math:`n`, with the convention for :math:`k` stated by
the result that produced it.

Two commitments shape the surface. First, the two records are defined
once, in ``_core``, and this module is their public name: every
``summary()`` and ``information_criteria`` in the package returns an
instance of the classes documented here, so their methods and fields are
stable API, and the engine pages document the same objects only as
internals. Second, nothing here is specific to a model family; a record
that needs more structure than a title, metadata, one table and notes
belongs beside the result that needs it.

Layout. :class:`SummaryTable` with its ``to_text``, ``to_html``,
``to_pandas`` and ``to_polars`` renderers, and :class:`InformationCriteria`
with ``from_likelihood``. Both live in ``_core._containers`` and are bound
to this module so that annotations, pickles and the API reference all
name them here.

Example:
    >>> import numpy as np
    >>> from cultivars.summary import InformationCriteria, SummaryTable
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> res = ARIMA(np.random.default_rng(0).standard_normal(120), order=(1, 0, 0)).fit()
    >>> isinstance(res.summary(), SummaryTable), isinstance(res.information_criteria, InformationCriteria)
    (True, True)
    >>> SummaryTable.__module__, InformationCriteria.__module__
    ('cultivars.summary', 'cultivars.summary')
"""

from .engine._core._containers import _InformationCriteria as InformationCriteria
from .engine._core._containers import _SummaryTable as SummaryTable

__all__ = ["InformationCriteria", "SummaryTable"]
