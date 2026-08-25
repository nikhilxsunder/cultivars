
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalLevels:
    """One unit's equations written in levels, ready to be linked.

    The common currency of a global system. Units may be estimated as a VARX in
    levels or as a conditional error-correction model in differences, and those
    are different objects with different parameters -- but both imply the same
    thing about how the unit's variables respond to their own past and to the
    foreign aggregates, and that implication is what the linkage consumes.
    Converting each unit to this record once means the global solve never learns
    which estimator produced which unit.

    Attributes:
        phi: ``(p, k, k)`` coefficients on the unit's own lagged levels.
        impact: ``(k, m)`` contemporaneous response to the foreign levels.
        exog_lags: ``(p, k, m)`` coefficients on lagged foreign levels, padded
            to the same depth as ``phi`` so the two stack row by row.
        deterministic: ``(d, k)`` deterministic coefficients.
        names: The unit's own variable labels.
        exog_names: The unit's foreign-aggregate labels.
    """

    phi: npt.NDArray[np.float64]
    impact: npt.NDArray[np.float64]
    exog_lags: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    names: tuple[str, ...]
    exog_names: tuple[str, ...]

    @property
    def order(self) -> int:
        """Lags of the unit's own levels."""
        return int(self.phi.shape[0])

    @property
    def k_endog(self) -> int:
        """Variables the unit models."""
        return int(self.phi.shape[1])

    @property
    def k_exog(self) -> int:
        """Foreign aggregates the unit reads."""
        return int(self.impact.shape[1])
