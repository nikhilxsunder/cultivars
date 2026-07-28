"""Public exception and warning hierarchy for cultivars.

All exceptions derive from :class:`CultivarsError`, and each also derives
from the builtin that best matches its failure mode so that user code
catching ``ValueError`` / ``RuntimeError`` continues to work.
"""

from __future__ import annotations


class CultivarsError(Exception):
    """Base class for every exception raised by cultivars."""


class DimensionError(CultivarsError, ValueError):
    """An array has the wrong shape, rank, or is non-conformable."""


class SpecificationError(CultivarsError, ValueError):
    """A model, polynomial, or coefficient set is invalid or inconsistent."""


class NumericalError(CultivarsError, RuntimeError):
    """A computation failed numerically (non-finite input, singular system)."""


class StabilityWarning(UserWarning):
    """Emitted by higher layers when a fitted model is non-stationary."""
