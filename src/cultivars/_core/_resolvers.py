from __future__ import annotations

from collections.abc import Sequence

from ..exceptions import SpecificationError


def _resolve_ordering(names: tuple[str, ...], order: Sequence[str] | None) -> tuple[int, ...]:
    """Map a declared variable ordering onto column indices.

    Args:
        names: The result's variable labels.
        order: The declared ordering, or ``None`` for the labels as given.

    Returns:
        A permutation of ``range(k)``.

    Raises:
        SpecificationError: If ``order`` is not a permutation of ``names``.
    """
    if order is None:
        return tuple(range(len(names)))
    declared = tuple(str(name) for name in order)
    if sorted(declared) != sorted(names):
        raise SpecificationError(
            f"order must be a permutation of the variable names {names}; got {declared}."
        )
    return tuple(names.index(name) for name in declared)
