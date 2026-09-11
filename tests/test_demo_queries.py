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
        ("fire safety requirements for a commercial building", "no_match"),
        ("quantum satellite launch licensing", "no_match"),
        ("industrial pumps", "no_match"),
    ),
)
def test_demo_queries_are_no_match_without_eligible_standards(
    query: str,
    expected_state: str,
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With catalog-only / zero-eligible officialBIS corpus, recommendations stay empty.

    Historical IS 5120 checklist data is not promoted into recommendations when an
    officialBIS-derived corpus exists without recommendation-eligible documents.
    """
    monkeypatch.setattr(
        app_module.qwen_reasoner,
        "explain",
        lambda *_args, **_kwargs: ExplanationOutcome(None),
    )
    response = app_module.app.test_client().post("/get_result", json={"user_input": query})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_state"] == expected_state
    assert payload["recommendations"] == []
    assert "No sufficiently supported" in payload["warnings"][0]
