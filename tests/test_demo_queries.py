from __future__ import annotations

import pytest
import importlib
import sys

from backend.qwen_reasoning import ExplanationOutcome


@pytest.fixture(scope="module")
def app_module():
    sys.modules.pop("backend.app", None)
    return importlib.import_module("backend.app")


@pytest.mark.parametrize(
    ("query", "expected_state"),
    (
        ("general-purpose centrifugal pumps for industrial water service", "recommendations"),
        ("centrifugal pumps", "recommendations"),
        ("induction motors", "no_match"),
        ("fire safety requirements for a commercial building", "no_match"),
        ("quantum satellite launch licensing", "no_match"),
    ),
)
def test_demo_queries_cover_recommendation_and_no_match_paths(
    query: str,
    expected_state: str,
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module.qwen_reasoner,
        "explain",
        lambda *_args, **_kwargs: ExplanationOutcome(None),
    )
    response = app_module.app.test_client().post("/get_result", json={"user_input": query})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_state"] == expected_state
    if expected_state == "recommendations":
        recommendation = payload["recommendations"][0]
        assert recommendation["standard_number"] == "IS 5120"
        assert recommendation["title"]
        assert recommendation["evidence"][0]["text"]
        assert recommendation["evidence"][0]["page_start"] == 1
        assert 0 <= recommendation["score"] <= 1
        assert recommendation["explanation_mode"] in {"generative", "deterministic"}
    else:
        assert payload["recommendations"] == []
        assert "No sufficiently supported" in payload["warnings"][0]


def test_broad_industrial_pumps_query_is_not_confidently_forced(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broad category-only wording must not become a confident recommendation."""
    monkeypatch.setattr(
        app_module.qwen_reasoner,
        "explain",
        lambda *_args, **_kwargs: ExplanationOutcome(None),
    )
    response = app_module.app.test_client().post("/get_result", json={"user_input": "industrial pumps"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_state"] == "no_match"
    assert payload["recommendations"] == []
