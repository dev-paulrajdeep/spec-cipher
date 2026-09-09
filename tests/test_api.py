from __future__ import annotations

import importlib
import sys
from threading import RLock
from types import SimpleNamespace

import pytest

from backend.qwen_reasoning import ExplanationOutcome
from intelligence.models import (
    ApplicabilityScore,
    ConstraintReason,
    ConstraintResult,
    EvidenceItem,
    Recommendation,
    RelatedStandard,
    RerankResult,
)
from intelligence.pipeline import IntelligenceEngine
from retrieval.service import RetrievalService


class FakeRetriever:
    def __init__(self) -> None:
        self._search_lock = RLock()
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> tuple[object, ...]:
        self.calls.append((query, top_k))
        return ()


class FakeReasoner:
    def __init__(self, outcome: ExplanationOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, Recommendation]] = []

    def explain(self, user_input: str, recommendation: Recommendation) -> ExplanationOutcome:
        self.calls.append((user_input, recommendation))
        return self.outcome


def recommendation() -> Recommendation:
    return Recommendation(
        standard_number="IS 2645:2003",
        title="Waterproofing compound",
        final_score=0.82,
        rank=1,
        constraint_result=ConstraintResult(True, (ConstraintReason("status_ok", "Current standard."),)),
        rerank_result=RerankResult(0.7, 0.7, 1, "token_overlap"),
        applicability=ApplicabilityScore(1.0, 0.8, 1.0, 0.9, 0.9, 0.92),
        evidence=(
            EvidenceItem(
                "IS 2645:2003",
                "chunk-1",
                "IS2645.pdf",
                4,
                5,
                "2.1",
                "Scope",
                "Direct source wording.",
                0.82,
            ),
        ),
        related_standards=(
            RelatedStandard(
                "IS 3370 (Part 1):2021",
                "references",
                "IS 2645:2003",
                6,
                "Related standard reference.",
                1,
            ),
        ),
        status="current",
        version="2003",
    )


def load_app(monkeypatch: pytest.MonkeyPatch, recommendations: tuple[Recommendation, ...]):
    retriever = FakeRetriever()

    def fake_engine_init(self, *args, **kwargs) -> None:
        return None

    def fake_recommend(self, user_input: str, results: tuple[object, ...]) -> SimpleNamespace:
        return SimpleNamespace(recommendations=recommendations)

    monkeypatch.setattr(
        RetrievalService,
        "from_persisted",
        classmethod(lambda cls, *args, **kwargs: retriever),
    )
    monkeypatch.setattr(IntelligenceEngine, "__init__", fake_engine_init)
    monkeypatch.setattr(IntelligenceEngine, "recommend", fake_recommend)
    sys.modules.pop("backend.app", None)
    module = importlib.import_module("backend.app")
    return module, retriever


def stable_recommendation_fields(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"explanation", "warnings"}
    }


def test_endpoint_adds_explanation_without_changing_recommendation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = recommendation()
    module, retriever = load_app(monkeypatch, (source,))
    explanation = {
        "summary": "Evidence supports the finalized recommendation.",
        "key_points": ["The scope text is relevant."],
        "caveats": ["Verify the complete standard before use."],
    }
    reasoner = FakeReasoner(ExplanationOutcome(explanation))
    module.qwen_reasoner = reasoner

    response = module.app.test_client().post(
        "/get_result",
        json={"user_input": "waterproofing compound for concrete"},
    )

    assert response.status_code == 200
    body = response.get_json()
    payload = body["recommendations"][0]
    assert stable_recommendation_fields(payload) == stable_recommendation_fields(source.to_dict())
    assert payload["warnings"] == source.to_dict()["warnings"]
    assert payload["explanation"] == explanation
    assert retriever.calls == [("waterproofing compound for concrete", 30)]
    assert reasoner.calls == [("waterproofing compound for concrete", source)]


def test_endpoint_fails_open_when_explanation_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = recommendation()
    module, _ = load_app(monkeypatch, (source,))
    module.qwen_reasoner = FakeReasoner(
        ExplanationOutcome(None, "Explanation unavailable: Ollama request timed out")
    )

    response = module.app.test_client().post("/get_result", json={"user_input": "waterproofing"})
    payload = response.get_json()["recommendations"][0]

    assert stable_recommendation_fields(payload) == stable_recommendation_fields(source.to_dict())
    assert payload["explanation"] is None
    assert payload["warnings"] == [
        "Explanation unavailable: Ollama request timed out"
    ]


def test_empty_recommendations_do_not_invoke_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = load_app(monkeypatch, ())
    reasoner = FakeReasoner(ExplanationOutcome({"summary": "unexpected", "key_points": [], "caveats": []}))
    module.qwen_reasoner = reasoner

    response = module.app.test_client().post("/get_result", json={"user_input": "nothing applicable"})

    assert response.status_code == 200
    assert response.get_json()["recommendations"] == []
    assert response.get_json()["warnings"] == [
        "No suitable standard was found for this requirement."
    ]
    assert reasoner.calls == []


def test_endpoint_caps_final_recommendations(monkeypatch: pytest.MonkeyPatch) -> None:
    source = recommendation()
    module, _ = load_app(monkeypatch, tuple(source for _ in range(7)))
    module.qwen_reasoner = FakeReasoner(ExplanationOutcome(None))

    response = module.app.test_client().post("/get_result", json={"user_input": "waterproofing"})

    assert response.status_code == 200
    assert len(response.get_json()["recommendations"]) == 5

