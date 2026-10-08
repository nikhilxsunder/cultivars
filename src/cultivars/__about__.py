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
r"""Package metadata: the single source of the version and the identity strings.

The build reads ``__version__`` from this file (``[tool.hatch.version]``
in ``pyproject.toml`` points here), so the distribution's version and
the installed package's agree by construction rather than by a release
checklist; bump it here and nowhere else. The remaining names are the
identity strings the documentation build and the distribution metadata
share -- title, one-line and long descriptions, author, licence,
repository and documentation URLs -- kept as plain module attributes so
that ``cultivars.__about__.__version__`` is readable without importing
anything that touches ``numpy`` or ``scipy``.

The version follows PEP 440. A pre-release tag (``1.0.0a1``) marks an
API that may still move; a final ``1.0.0`` is the point after which
public names in the leaf modules are stable and removals go through a
deprecation cycle.

Example:
    >>> import re
    >>> from cultivars.__about__ import __version__, __title__, __license__
    >>> __title__, __license__
    ('cultivars', 'MIT')
    >>> bool(re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", __version__))
    True
"""

__title__ = "cultivars"
__description__ = (
    "Research-grade time series econometrics: ARIMA to state space, VAR to structural"
    " identification, Bayesian and spectral methods."
)
__summary__ = "Research-grade time series econometrics."
__version__ = "1.0.0a2"
__copyright__ = "Copyright (c) 2026 Nikhil Sunder"
__author__ = "Nikhil Sunder"
__email__ = "nsunder724@gmail.com"
__license__ = "MIT"
__repository__ = "https://github.com/nikhilxsunder/cultivars"
__docs__ = "https://nikhilxsunder.github.io/cultivars"
