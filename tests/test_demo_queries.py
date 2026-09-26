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
        ("general-purpose centrifugal pumps for industrial water service", "no_match"),
        ("centrifugal pumps", "no_match"),
        ("induction motors", "no_match"),
        ("fire safety requirements for a commercial building", "recommendations"),
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
    """Only queries aligned with the current officialBIS-derived standard are recommended."""
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
        assert payload["recommendations"][0]["standard_number"] == "SP 7:2026"
