"""Unit tests for Settings, ApplicationContainer, and create_app factory."""

from __future__ import annotations

from typing import Any

import pytest

from speccipher.api.app import create_app
from speccipher.config.settings import ServerSettings, Settings
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk
from speccipher.domain.recommendation import Confidence, Recommendation
from speccipher.services.container import ApplicationContainer


class DummyRetriever:
    def __init__(self, ready: bool = True) -> None:
        self.is_ready = ready
        self.closed = False
        self.search_calls: list[tuple[str, int]] = []

    def readiness(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "model_loaded": self.is_ready,
            "embedding_model": "dummy-bge-m3",
            "embedding_device": "cpu",
            "chunk_count": 42,
            "standard_count": 5,
        }

    def search(self, query: str, top_k: int = 500) -> tuple[Any, ...]:
        self.search_calls.append((query, top_k))
        return ()

    def close(self) -> None:
        self.closed = True


class DummyDecisionEngine:
    def __init__(self, recs: tuple[Recommendation, ...] = ()) -> None:
        self.recs = recs

    def recommend(self, user_input: str, results: tuple[Any, ...]) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(recommendations=self.recs, candidates=())


class DummyReasoner:
    def explain(self, user_input: str, recommendation: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            explanation={"summary": "Dummy summary", "key_points": [], "caveats": []},
            warning=None,
            fallback_reason=None,
            mode="generative",
        )


class TestSettings:
    def test_default_settings(self) -> None:
        s = Settings.from_env({})
        assert s.server.host == "127.0.0.1"
        assert s.server.port == 5000
        assert s.server.max_input_characters == 4000
        assert s.retrieval.embedding_model == "BAAI/bge-m3"
        assert s.explanation.ollama_host == "http://localhost:11434"

    def test_legacy_qwen_aliases(self) -> None:
        env = {
            "QWEN_OLLAMA_HOST": "http://qwen-host:11434",
            "QWEN_MODEL_NAME": "qwen-special:latest",
            "QWEN_TIMEOUT_SECONDS": "90.0",
            "QWEN_KEEP_ALIVE": "5",
            "QWEN_THINK": "1",
        }
        s = Settings.from_env(env)
        assert s.explanation.ollama_host == "http://qwen-host:11434"
        assert s.explanation.model_name == "qwen-special:latest"
        assert s.explanation.timeout_seconds == 90.0
        assert s.explanation.keep_alive == 5
        assert s.explanation.think is True

    def test_server_settings_validation(self) -> None:
        with pytest.raises(ValueError):
            ServerSettings(port=99999)
        with pytest.raises(ValueError):
            ServerSettings(max_content_length=500)


class TestApplicationContainer:
    def test_zero_heavy_init_on_instantiation(self) -> None:
        c = ApplicationContainer()
        assert c._retrieval_service is None
        assert c._decision_engine is None
        assert c._explanation_service is None
        readiness = c.readiness()
        assert readiness["ready"] is False

    def test_dependency_injection_overrides(self) -> None:
        retriever = DummyRetriever()
        decision = DummyDecisionEngine()
        reasoner = DummyReasoner()

        c = ApplicationContainer(
            retrieval_override=retriever,
            decision_override=decision,
            explanation_override=reasoner,
        )
        assert c.retrieval_service is retriever
        assert c.decision_engine is decision
        assert c.intelligence_engine is decision
        assert c.explanation_service is reasoner
        assert c.qwen_reasoner is reasoner

        readiness = c.readiness()
        assert readiness["ready"] is True
        assert readiness["retriever_loaded"] is True

    def test_close_and_context_manager(self) -> None:
        retriever = DummyRetriever()
        with ApplicationContainer(retrieval_override=retriever) as c:
            assert c.readiness()["ready"] is True
        assert retriever.closed is True
        assert c._is_closed is True


class TestAppFactory:
    def test_create_app_endpoints(self) -> None:
        retriever = DummyRetriever()
        decision = DummyDecisionEngine()
        reasoner = DummyReasoner()
        container = ApplicationContainer(
            retrieval_override=retriever,
            decision_override=decision,
            explanation_override=reasoner,
        )
        app = create_app(container)
        client = app.test_client()

        # Health live
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "live"
        assert "X-Request-ID" in resp.headers

        # Health ready
        resp_ready = client.get("/health/ready")
        assert resp_ready.status_code == 200
        assert resp_ready.get_json()["ready"] is True

        # Legacy health
        resp_legacy_health = client.get("/health")
        assert resp_legacy_health.status_code == 200
        assert resp_legacy_health.get_json()["status"] == "ready"

    def test_get_result_endpoint_flow(self) -> None:
        sid = StandardId.parse("IS 456:2000")
        chunk = DocumentChunk(
            chunk_id="c-1",
            standard_id=sid,
            logical_doc_id="doc-1",
            source_file_id="IS456.pdf",
            absolute_page_start=14,
            absolute_page_end=15,
            section_number="5.1",
            section_title="General",
            text="Concrete specifications.",
        )
        rec = Recommendation(
            standard_id=sid,
            title="Plain and Reinforced Concrete",
            rank=1,
            confidence=Confidence(level="High", score=0.9),
            similarity_score=0.85,
            applicability_score=0.95,
            evidence=(chunk,),
        )
        retriever = DummyRetriever()
        decision = DummyDecisionEngine((rec,))
        reasoner = DummyReasoner()
        container = ApplicationContainer(
            retrieval_override=retriever,
            decision_override=decision,
            explanation_override=reasoner,
        )
        app = create_app(container)
        client = app.test_client()

        resp = client.post("/get_result", json={"user_input": "concrete specification"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["result_state"] == "recommendations"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["is_code"] == "IS 456:2000"
        assert data["recommendations"][0]["explanation_mode"] == "generative"
        assert retriever.search_calls == [("concrete specification", 500)]
