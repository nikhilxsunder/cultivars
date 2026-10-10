"""Tests for structural sign-restriction exports."""

from __future__ import annotations

from cultivars.multivariate.structural import sign_restrictions


def test_narrative_sign_restricted_svar_is_exported() -> None:
    """Test that the narrative sign-restriction identification class is exported."""
    assert "NarrativeSignRestrictedSVAR" in sign_restrictions.__all__
