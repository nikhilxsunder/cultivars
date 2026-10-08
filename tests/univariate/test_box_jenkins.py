# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Regression tests for ARMA specification validation."""

from __future__ import annotations

import numpy as np
import pytest

from cultivars.exceptions import SpecificationError
from cultivars.univariate.box_jenkins import ARMA


@pytest.mark.parametrize("order", [(), (1,), (1, 0, 1)])
def test_arma_rejects_wrong_order_length(order: tuple[int, ...]) -> None:
    """Wrong arity raises the documented library exception before unpacking."""
    with pytest.raises(SpecificationError, match=r"order must have 2 elements.*p.*q"):
        ARMA(np.arange(50.0), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("order", "label"),
    [((-1, 0), "p"), ((0, -1), "q"), ((True, 0), "p"), ((0, 1.5), "q")],
)
def test_arma_rejects_invalid_order_terms(order: tuple[int, int], label: str) -> None:
    """Keep named validation for negative and non-integral ARMA terms."""
    with pytest.raises(SpecificationError, match=rf"{label} must be"):
        ARMA(np.arange(50.0), order=order)


@pytest.mark.parametrize("order", [(0, 0), (1, 0), (0, 1), (2, 1)])
def test_arma_preserves_valid_specification(order: tuple[int, int]) -> None:
    """Valid orders still expand into the same non-seasonal specification."""
    model = ARMA(np.arange(50.0), order=order, trend="n")

    assert model.order == (order[0], 0, order[1])
    assert model.seasonal_order == (0, 0, 0, 0)
    assert model.trend == "n"
