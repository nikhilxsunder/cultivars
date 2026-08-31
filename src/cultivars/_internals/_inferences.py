
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._results import _WaldTestResult


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

    def wald(self, cells: Sequence[tuple[int, int]], *, null: str) -> _WaldTestResult:
        """Test that a set of coefficients is jointly zero."""
        ...

    def wald_restriction(
        self, restriction: npt.NDArray[np.float64], *, null: str
    ) -> _WaldTestResult:
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