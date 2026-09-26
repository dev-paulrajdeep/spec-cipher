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
        ("building design considering wind effects on occupant comfort inside and outside", "recommendations"),
        ("general-purpose centrifugal pumps for industrial water service", "no_match"),
        ("centrifugal pumps", "no_match"),
        ("induction motors", "no_match"),
        ("quantum satellite launch licensing", "no_match"),
        ("industrial pumps", "no_match"),
    ),
)
def test_demo_queries_respect_source_corpus_applicability(
    query: str,
    expected_state: str,
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only supported queries for the current officialBIS-derived corpus are recommended."""
    monkeypatch.setattr(
        app_module.qwen_reasoner,
        "explain",
        lambda *_args, **_kwargs: ExplanationOutcome(None),
    )
    response = app_module.app.test_client().post("/get_result", json={"user_input": query})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_state"] == expected_state
    if expected_state == "no_match":
        assert payload["recommendations"] == []
        assert "No sufficiently supported" in payload["warnings"][0]
    else:
        assert payload["recommendations"]
        recommendation = payload["recommendations"][0]
        assert recommendation["standard_number"] == "SP 7:2026"
        assert recommendation["title"] == "NATIONAL BUILDING CONSTRUCTION"
        evidence = recommendation["evidence"]
        assert evidence
        assert evidence[0]["section_number"] == "4.1.11"
        assert evidence[0]["page_start"] == 185
        assert evidence[0]["page_end"] == 186
        assert "effects of wind on the comfort of people inside" in evidence[0]["text"].lower()
