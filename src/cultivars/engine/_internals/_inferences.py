# filepath: /src/cultivars/engine/_internals/_inferences.py
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
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._tests import _WaldTest

__all__ = [
    "_CoefficientInference",
]


@runtime_checkable
class _CoefficientInference(Protocol):
    """What every result needs from whatever produced its coefficients.

    Two objects satisfy this and they answer the same questions from different
    theories: :class:`_CoefficientCovariance` from a sampling distribution,
    :class:`_PosteriorCovariance` from a posterior. Naming the interface rather
    than the union is what keeps the mixin from having to know which it has --
    and ``_IS_POSTERIOR`` is here precisely so the one place that *must* know,
    the coefficient table's headings, can ask.
    """

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """The estimate these moments describe, in design-column order."""
        ...

    @property
    def _IS_POSTERIOR(self) -> bool:
        """Whether these are posterior moments rather than sampling moments."""
        ...

    @property
    def stderr(self) -> npt.NDArray[np.float64]:
        """Standard deviations, shaped like the coefficients."""
        ...

    @property
    def tstat(self) -> npt.NDArray[np.float64]:
        """Coefficients over their standard deviations."""
        ...

    @property
    def pvalue(self) -> npt.NDArray[np.float64]:
        """Two-sided normal tail areas."""
        ...

    def conf_int(self, *, alpha: float = 0.05) -> tuple[npt.NDArray[np.float64], ...]:
        """Symmetric bounds at the given level."""
        ...

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """The full covariance over ``vec(B)``."""
        ...

    def wald(self, cells: Sequence[tuple[int, int]], *, null: str) -> _WaldTest:
        """Test that a set of coefficients is jointly zero."""
        ...

    def wald_restriction(self, restriction: npt.NDArray[np.float64], *, null: str) -> _WaldTest:
        """Test a general set of linear restrictions."""
        ...

    @property
    def effective_parameters(self) -> float:
        """Parameters the coefficient estimate actually spends.

        Nominal for an unrestricted fit and strictly smaller under shrinkage,
        which is what makes it worth asking for uniformly: an information
        criterion that charges the nominal width penalizes a shrunk model for
        freedom it never used, and a caller comparing a shrunk fit against an
        unshrunk one has no way to notice unless both answer the same question.
        """
        ...
