from __future__ import annotations

import json

import pytest

from scripts.e2e_smoke import (
    DEFAULT_NO_RESULT_QUERY,
    DEFAULT_QUERY,
    SmokeFailure,
    rendered_score,
    stable_recommendation,
    verify_index,
)


def test_smoke_defaults_and_recommendation_projection() -> None:
    assert DEFAULT_QUERY == "fire safety requirements for residential buildings"
    assert DEFAULT_NO_RESULT_QUERY.startswith("IS 99999")
    payload = {
        "standard_number": "IS 1234:2026",
        "title": "Safety standard",
        "final_score": 0.8,
        "warnings": ["optional warning"],
        "explanation": {"summary": "summary", "key_points": [], "caveats": []},
    }
    assert stable_recommendation(payload) == {
        "standard_number": "IS 1234:2026",
        "title": "Safety standard",
        "final_score": 0.8,
    }


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.91, "91%"), (1, "100%"), (1.25, "1.25"), (None, "-")],
)
def test_rendered_score_matches_frontend_format(score: object, expected: str) -> None:
    assert rendered_score(score) == expected


def test_verify_index_reports_missing_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.e2e_smoke.ROOT", tmp_path)
    (tmp_path / "retrieval-index").mkdir()
    (tmp_path / "retrieval-index" / "index.faiss").write_bytes(b"index")
    with pytest.raises(SmokeFailure, match="missing"):
        verify_index()
