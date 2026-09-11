from __future__ import annotations

import importlib
import json

import pytest
import requests

from intelligence.models import (
    ApplicabilityScore,
    ConstraintReason,
    ConstraintResult,
    EvidenceItem,
    Recommendation,
    RelatedStandard,
    RerankResult,
)
from backend.qwen_reasoning import (
    DEFAULT_QWEN_MODEL_NAME,
    DEFAULT_QWEN_OLLAMA_HOST,
    DEFAULT_QWEN_TIMEOUT_SECONDS,
    EXPLANATION_FORMAT,
    MAX_EVIDENCE_TEXT_CHARACTERS,
    MAX_PROJECTION_CHARACTERS,
    MAX_RELATED_STANDARDS,
    QwenConfig,
    QwenReasoner,
    SYSTEM_INSTRUCTION,
)


class FakeResponse:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error

    def json(self) -> object:
        return self._payload


def recommendation(
    *,
    title: str | None = "Waterproofing compound",
    status: str | None = "current",
    evidence_text: str = "Direct source wording.",
    evidence: tuple[EvidenceItem, ...] | None = None,
    related_standards: tuple[RelatedStandard, ...] | None = None,
) -> Recommendation:
    return Recommendation(
        standard_number="IS 2645:2003",
        title=title,
        final_score=0.82,
        rank=1,
        constraint_result=ConstraintResult(True, (ConstraintReason("status_ok", "Current standard."),)),
        rerank_result=RerankResult(0.7, 0.7, 1, "token_overlap"),
        applicability=ApplicabilityScore(1.0, 0.8, 1.0, 0.9, 0.9, 0.92),
        evidence=evidence if evidence is not None else (
            EvidenceItem(
                "IS 2645:2003",
                "chunk-1",
                "IS2645.pdf",
                4,
                5,
                "2.1",
                "Scope",
                evidence_text,
                0.82,
            ),
        ),
        related_standards=related_standards if related_standards is not None else (
            RelatedStandard(
                "IS 3370 (Part 1):2021",
                "references",
                "IS 2645:2003",
                6,
                "Related standard reference.",
                1,
            ),
        ),
        status=status,
        version="2003",
    )


def valid_content() -> str:
    return json.dumps(
        {
            "summary": "The recommendation is supported by the supplied evidence.",
            "key_points": ["The supplied scope text addresses waterproofing."],
            "caveats": ["Verify the complete standard before use."],
        }
    )


def test_environment_configuration_uses_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("QWEN_MODEL_NAME", raising=False)
    monkeypatch.delenv("QWEN_TIMEOUT_SECONDS", raising=False)
    assert QwenConfig.from_environment() == QwenConfig(
        DEFAULT_QWEN_OLLAMA_HOST,
        DEFAULT_QWEN_MODEL_NAME,
        DEFAULT_QWEN_TIMEOUT_SECONDS,
    )

    monkeypatch.setenv("QWEN_OLLAMA_HOST", "http://ollama.example:22123/")
    monkeypatch.setenv("QWEN_MODEL_NAME", "team-qwen:latest")
    monkeypatch.setenv("QWEN_TIMEOUT_SECONDS", "12.5")
    assert QwenConfig.from_environment() == QwenConfig(
        "http://ollama.example:22123",
        "team-qwen:latest",
        12.5,
    )


def test_importing_module_does_not_contact_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: calls.append((args, kwargs)))
    import backend.qwen_reasoning as qwen_reasoning

    importlib.reload(qwen_reasoning)

    assert calls == []


def test_projection_is_deterministic_and_contains_only_finalized_fields() -> None:
    reasoner = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3), post=lambda **_: None)
    first = reasoner.project_recommendation("waterproofing requirement", recommendation())
    second = reasoner.project_recommendation("waterproofing requirement", recommendation())

    assert first == second
    assert set(first) == {
        "user_input",
        "standard_number",
        "title",
        "status",
        "version",
        "final_score",
        "constraints",
        "applicability",
        "evidence",
        "related_standards",
    }
    assert first["evidence"][0] == {
        "standard_number": "IS 2645:2003",
        "chunk_id": "chunk-1",
        "source_file": "IS2645.pdf",
        "page_start": 4,
        "page_end": 5,
        "section_number": "2.1",
        "section_title": "Scope",
        "text": "Direct source wording.",
        "relevance_score": 0.82,
    }


def test_projection_keeps_missing_metadata_null_without_invention() -> None:
    projection = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3)).project_recommendation(
        "requirement",
        recommendation(title=None, status=None, evidence=()),
    )

    assert projection["title"] is None
    assert projection["status"] is None
    assert projection["evidence"] == []


def test_projection_truncates_oversized_evidence_text() -> None:
    projection = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3)).project_recommendation(
        "requirement",
        recommendation(evidence_text="e" * 2_000),
    )

    assert len(projection["evidence"][0]["text"]) == MAX_EVIDENCE_TEXT_CHARACTERS
    assert projection["evidence"][0]["text"].endswith("... [truncated]")


def test_projection_bounds_combined_evidence_budget() -> None:
    evidence = tuple(
        EvidenceItem(
            "IS 2645:2003",
            f"chunk-{index}",
            "IS2645.pdf",
            4,
            5,
            "2.1",
            "Scope",
            "e" * 500,
            0.82,
        )
        for index in range(20)
    )

    projection = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3)).project_recommendation(
        "requirement",
        recommendation(evidence=evidence),
    )

    serialized = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert len(serialized) <= MAX_PROJECTION_CHARACTERS
    assert len(projection["evidence"][0]["text"]) < MAX_EVIDENCE_TEXT_CHARACTERS


def test_projection_bounds_large_relationship_graph() -> None:
    related = tuple(
        RelatedStandard(
            f"IS {index}:2026",
            "references",
            "SP 7:2026",
            index,
            "relationship evidence " * 100,
            1,
        )
        for index in range(MAX_RELATED_STANDARDS + 100)
    )

    projection = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3)).project_recommendation(
        "requirement",
        recommendation(related_standards=related),
    )

    assert len(projection["related_standards"]) == MAX_RELATED_STANDARDS
    assert len(json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) <= MAX_PROJECTION_CHARACTERS


def test_request_uses_configured_ollama_schema_keep_alive_and_final_content_only() -> None:
    captured: dict[str, object] = {}

    def post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "message": {
                    "content": valid_content(),
                    "thinking": "private chain of thought",
                }
            }
        )

    reasoner = QwenReasoner(QwenConfig("http://ollama.test:1234/", "qwen-team:latest", 7), post=post)
    outcome = reasoner.explain("waterproofing requirement", recommendation())

    assert captured["url"] == "http://ollama.test:1234/api/chat"
    assert captured["timeout"] == 7
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["model"] == "qwen-team:latest"
    assert body["stream"] is False
    assert body["keep_alive"] == 0
    assert body["format"] == EXPLANATION_FORMAT
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_INSTRUCTION}
    assert outcome.explanation == json.loads(valid_content())
    assert "private chain of thought" not in json.dumps(outcome.explanation)
    assert outcome.warning is None


@pytest.mark.parametrize(
    "payload",
    [
        {"message": {"content": json.dumps({"summary": "x", "key_points": []})}},
        {"message": {"content": "not json"}},
        {"message": {"content": ""}},
        {},
    ],
)
def test_invalid_or_incomplete_responses_fail_open_without_retry(payload: object) -> None:
    calls = 0

    def post(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(payload)

    outcome = QwenReasoner(QwenConfig("http://ollama.test", "model", 3), post=post).explain(
        "requirement",
        recommendation(),
    )

    assert calls == 1
    assert outcome.explanation is not None
    assert "IS 2645:2003" in outcome.explanation["summary"]
    assert outcome.warning == "AI explanation unavailable; showing a deterministic evidence summary."


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("refused"),
        requests.Timeout("late"),
        MemoryError("out of memory"),
    ],
)
def test_transport_and_memory_failures_use_deterministic_fallback(failure: Exception) -> None:
    def post(*args: object, **kwargs: object) -> FakeResponse:
        raise failure

    outcome = QwenReasoner(QwenConfig("http://ollama.test", "model", 3), post=post).explain(
        "requirement",
        recommendation(),
    )

    assert outcome.explanation is not None
    assert outcome.explanation["key_points"]
    assert outcome.warning == "AI explanation unavailable; showing a deterministic evidence summary."


def test_messages_include_centralized_grounding_instruction() -> None:
    reasoner = QwenReasoner(config=QwenConfig("http://ollama.test", "model", 3))
    messages = reasoner.build_messages(reasoner.project_recommendation("requirement", recommendation()))

    assert messages[0]["content"] == SYSTEM_INSTRUCTION
    assert "Do not invent standards" in messages[0]["content"]

