# filepath: /src/cultivars/experimental/__init__.py
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
r"""Recent models that are not yet packaged anywhere, under a looser contract.

The stable subpackages promise that a public name keeps its meaning
across minor releases. This one does not: it exists so that a model
from a recent paper can be used, tested against its published results,
and criticized before it earns that promise, and the price of that is
that anything here may change or disappear in any minor release.
Importing the package says so once, through
:class:`~cultivars.exceptions.ExperimentalWarning`; filter that warning
with :mod:`warnings` if you have read this page and accept the terms.

The rules a model here follows. It is not imported by the top-level
package, so ``import cultivars`` never pays for it. It documents what
has been verified and what has not: which fixture it reproduces, and
which claims of its paper remain unchecked. It keeps the model-fit-
result design of the stable packages, so a fitted result is a frozen
record with a ``summary`` whose notes state its assumptions, and its
errors derive from ``ValueError`` and ``RuntimeError`` like every other
error the package raises. And it moves out on a schedule: once its
replication fixture lands and its surface has held for two minor
releases, it is promoted to its permanent subpackage under a stable
name; four minor releases without promotion and it is removed. The
:doc:`roadmap <project/roadmap>` lists what is here, what is next, and
when each entry was admitted.

What is relaxed is the dependency contract, not the design. The stable
packages run on numpy and scipy alone; a model here may be estimated by
a library the core does not carry, in which case it lives in its own
distribution and this package re-exports it, under a prefix that keeps
its records distinct from the core's. A re-exported result is not a
cultivars result: the consumers written against the core's protocols
(identification, backtests, spectral views) do not accept it, and the
page of each re-export says what does carry across.

Layout. :mod:`~cultivars.experimental.spectral` re-exports the
Non-Stationary Spectral Decomposition Network from the ``ns-sdn``
distribution, a neural state-space model that reads a scalar series as
a trend plus sinusoids whose amplitude, frequency and participation
drift with a latent state.

Example:
    The warning is raised once, at import, and names its own class so
    that it can be silenced precisely:

    >>> import warnings
    >>> from cultivars.exceptions import ExperimentalWarning
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", ExperimentalWarning)
    ...     from cultivars.experimental.spectral import NSSDN
    >>> NSSDN.__name__
    'NSSDN'
"""

from __future__ import annotations

import warnings

from cultivars_engine.exceptions import ExperimentalWarning

from . import spectral

__all__ = ["spectral"]

warnings.warn(
    "cultivars.experimental is unstable: its API may change in any minor release.",
    ExperimentalWarning,
    stacklevel=2,
)
