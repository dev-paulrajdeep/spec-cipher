from __future__ import annotations

from types import SimpleNamespace

from speccipher.api.app import create_app
from speccipher.domain.candidates import Candidate
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk
from speccipher.services.container import ApplicationContainer


class _Catalog:
    def get_standard_by_code(self, code: str):
        return {
            "code": code,
            "family": "IS",
            "number": "456",
            "year": 2000,
            "title": "Concrete structures",
            "is_stub": 0,
            "is_eligible": 1,
            "status": "current",
        }

    def get_relationships(self, code: str):
        return []

    def get_scope_text(self, code: str):
        return {
            "standard_code": code,
            "scope": "This standard covers concrete structures.",
            "exclusions": "",
        }

    def get_chunks_by_ids(self, chunk_ids: list[str]):
        return {
            chunk_id: {
                "chunk_id": chunk_id,
                "logical_doc_id": "pub_is456_2000",
                "standard_code": "IS 456:2000",
                "source_file_id": "source-1",
                "source_file": "official-is456.pdf",
                "absolute_page_start": 12,
                "absolute_page_end": 13,
                "section_number": "4.1",
                "section_title": "Scope",
                "text": "Verified source wording from the release.",
                "is_eligible": 1,
            }
            for chunk_id in chunk_ids
        }


class _HybridRetrieval:
    def __init__(self, candidates: tuple[Candidate, ...]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, limit: int):
        self.queries.append((query, limit))
        return self.candidates


class _ServingDecisionEngine:
    def analyze(self, requirement: str, candidates: tuple[Candidate, ...]):
        return SimpleNamespace(
            decisions=(
                SimpleNamespace(
                    standard_code="IS 456:2000",
                    candidates=candidates,
                ),
            )
        )


def _explanation_retrieval(chunk_id: str = "chunk-verified") -> _HybridRetrieval:
    return _HybridRetrieval(
        (
            _candidate(
                chunk_id,
                dense_score=0.8,
                lexical_score=-1.0,
                rrf_score=0.03,
            ),
        )
    )


def _candidate(
    chunk_id: str,
    *,
    dense_score: float | None,
    lexical_score: float | None,
    rrf_score: float,
) -> Candidate:
    return Candidate(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            standard_id=StandardId.parse("IS 456:2000"),
            logical_doc_id="pub_is456_2000",
            source_file_id="is456.pdf",
            absolute_page_start=12,
            absolute_page_end=13,
            section_number="4.1",
            section_title="Scope",
            text="Concrete structures shall meet the stated requirements.",
        ),
        dense_score=dense_score,
        lexical_score=lexical_score,
        rrf_score=rrf_score,
    )


def test_explain_does_not_claim_generation_without_provider() -> None:
    container = ApplicationContainer(
        explanation_override=object(),
        hybrid_retrieval_override=_explanation_retrieval(),
        serving_decision_override=_ServingDecisionEngine(),
        catalog_override=_Catalog(),
    )
    client = create_app(container).test_client()

    response = client.post(
        "/v1/explain",
        json={
            "requirement": "concrete structures",
            "standard_id": "IS 456:2000",
            "evidence": [
                {
                    "chunk_id": "chunk-verified",
                    "section_number": "fake-section",
                    "text": "Caller supplied evidence must be ignored.",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] == "deterministic"
    assert "Generative explanation unavailable" in payload["warning"]
    assert "official-is456.pdf" in payload["explanation"]["summary"]
    assert payload["evidence_chunk_ids"] == ["chunk-verified"]
    assert "Caller supplied evidence" not in payload["explanation"]["summary"]


def test_explain_rejects_evidence_from_a_different_standard() -> None:
    class MismatchedEvidenceCatalog(_Catalog):
        def get_chunks_by_ids(self, chunk_ids: list[str]):
            records = super().get_chunks_by_ids(chunk_ids)
            for record in records.values():
                record["standard_code"] = "IS 875:2015"
            return records

    container = ApplicationContainer(
        explanation_override=object(),
        hybrid_retrieval_override=_explanation_retrieval(),
        serving_decision_override=_ServingDecisionEngine(),
        catalog_override=MismatchedEvidenceCatalog(),
    )
    response = create_app(container).test_client().post(
        "/v1/explain",
        json={
            "requirement": "concrete structures",
            "standard_id": "IS 456:2000",
            "evidence": [{"chunk_id": "chunk-verified"}],
        },
    )
    assert response.status_code == 422


def test_explain_rejects_standard_not_recommended_for_requirement() -> None:
    container = ApplicationContainer(
        explanation_override=object(),
        hybrid_retrieval_override=_explanation_retrieval(),
        serving_decision_override=_ServingDecisionEngine(),
        catalog_override=_Catalog(),
    )
    response = create_app(container).test_client().post(
        "/v1/explain",
        json={
            "requirement": "concrete structures",
            "standard_id": "IS 875:2015",
            "evidence": [{"chunk_id": "chunk-verified"}],
        },
    )
    assert response.status_code == 422


def test_explain_provider_receives_only_release_verified_evidence() -> None:
    class CapturingProvider:
        received = None

        def explain_verified_evidence(self, requirement: str, **kwargs):
            self.received = (requirement, kwargs)
            return SimpleNamespace(
                explanation={"summary": "Grounded explanation.", "key_points": [], "caveats": []},
                mode="generative",
                warning=None,
            )

    provider = CapturingProvider()
    container = ApplicationContainer(
        explanation_override=provider,
        hybrid_retrieval_override=_explanation_retrieval(),
        serving_decision_override=_ServingDecisionEngine(),
        catalog_override=_Catalog(),
    )
    response = create_app(container).test_client().post(
        "/v1/explain",
        json={
            "requirement": "concrete structures",
            "standard_id": "IS 456:2000",
            "evidence": [
                {
                    "chunk_id": "chunk-verified",
                    "source_file": "forged.pdf",
                    "text": "Forged caller text.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.get_json()["mode"] == "generative"
    assert provider.received[0] == "concrete structures"
    assert provider.received[1]["evidence"][0]["source_file"] == "official-is456.pdf"
    assert provider.received[1]["evidence"][0]["text"] == "Verified source wording from the release."


def test_analyze_uses_hybrid_candidates_and_keeps_scores_separate() -> None:
    hybrid = _HybridRetrieval(
        (
            _candidate("chunk-a", dense_score=0.82, lexical_score=-1.4, rrf_score=0.032),
            _candidate("chunk-b", dense_score=None, lexical_score=-0.7, rrf_score=0.016),
        )
    )
    container = ApplicationContainer(
        hybrid_retrieval_override=hybrid,
        catalog_override=_Catalog(),
    )
    client = create_app(container).test_client()

    response = client.post(
        "/v1/analyze",
        json={"requirement": "concrete structures", "options": {"top_k": 2}},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert hybrid.queries == [("concrete structures", 50)]
    understanding = payload["requirement_understanding"]
    assert understanding["raw_requirement"] == "concrete structures"
    assert understanding["material"] == ["concrete"]
    assert understanding["completeness"] == 0.5
    assert understanding["missing"] == ["application", "conditions"]
    recommendation = payload["recommendations"][0]
    assert recommendation["role"] == "primary"
    assert recommendation["score_breakdown"]["dense_similarity"] == 0.82
    assert recommendation["score_breakdown"]["lexical_bm25"] == -1.4
    assert recommendation["score_breakdown"]["rrf"] == 0.032
    assert recommendation["score_breakdown"]["rerank"] > 0
    assert recommendation["score_breakdown"]["applicability"] > 0
    assert recommendation["confidence"] == {
        "level": "HIGH",
        "reasons": ["APPLICABILITY_CONFIRMED"],
    }
    assert recommendation["evidence"][1]["dense_similarity"] is None


def test_analyze_rejects_invalid_top_k() -> None:
    container = ApplicationContainer(
        hybrid_retrieval_override=_HybridRetrieval(()),
        catalog_override=_Catalog(),
    )
    response = (
        create_app(container)
        .test_client()
        .post(
            "/v1/analyze",
            json={"requirement": "concrete", "options": {"top_k": True}},
        )
    )
    assert response.status_code == 400


def test_analyze_returns_reason_codes_when_catalog_constraints_reject_all_candidates() -> None:
    class ExcludingCatalog(_Catalog):
        def get_scope_text(self, code: str):
            return {
                "standard_code": code,
                "scope": "",
                "exclusions": "This standard does not apply to concrete structures.",
            }

    container = ApplicationContainer(
        hybrid_retrieval_override=_HybridRetrieval(
            (_candidate("chunk-a", dense_score=0.82, lexical_score=-1.4, rrf_score=0.032),)
        ),
        catalog_override=ExcludingCatalog(),
    )
    response = (
        create_app(container)
        .test_client()
        .post(
            "/v1/analyze",
            json={"requirement": "concrete structures"},
        )
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["result_state"] == "no_match"
    assert payload["recommendations"] == []
    assert payload["reason_codes"] == ["CONTRADICTORY_EVIDENCE", "NO_ELIGIBLE_MATCH"]
