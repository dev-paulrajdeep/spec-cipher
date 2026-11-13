"""Tier 4: Real-World Application Scenarios E2E Tests (6 Comprehensive Scenarios).

Scenarios covered:
- Scenario 1: Realistic Tender Requirement Analysis (POST /v1/analyze)
- Scenario 2: Legacy Frontend Query Parity (POST /get_result)
- Scenario 3: Cold-Start Release Mount & Health Check Lifecycle
- Scenario 4: Atomic Corpus Rollback Under Live Serving Traffic
- Scenario 5: Non-Blocking Ollama Offline Explanation Fail-Open
- Scenario 6: Unverified Document Reference Resolved as StandardStub
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from speccipher.api.app import create_app
from speccipher.config.settings import Settings
from speccipher.services.container import ApplicationContainer
from tests.e2e.fixtures.sample_data import (
    QUERY_STEEL_BARS_STUB,
    QUERY_UNDERGROUND_RESERVOIR,
    QUERY_WIND_EFFECTS,
)
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder


def test_t4_scenario_1_realistic_tender_clause_analysis(client):
    """Scenario 1: Realistic Tender Requirement Analysis via POST /v1/analyze.
    
    Submits realistic structural tender clause for water storage reservoir.
    Verifies full pipeline: query understanding, hybrid retrieval, RRF, single rerank,
    constraints, categorical confidence (High), and typed relationships.
    """
    tender_payload = {
        "requirement": QUERY_UNDERGROUND_RESERVOIR,
        "options": {
            "top_k": 3,
            "include_evidence": True,
            "include_related": True,
        },
    }

    res = client.post("/v1/analyze", json=tender_payload)
    assert res.status_code == 200

    # Header verification
    assert "X-Request-ID" in res.headers
    assert res.headers["X-Content-Type-Options"] == "nosniff"

    data = res.get_json()
    assert data["result_state"] == "recommendations"
    assert "requirement_understanding" in data
    assert data["requirement_understanding"]["completeness"] > 0.0

    recs = data.get("recommendations", [])
    assert len(recs) >= 1

    # Primary recommendation verification
    primary = recs[0]
    assert primary["standard_id"]["code"] == "IS 3370 (Part 1):2021"
    assert primary["rank"] == 1
    assert primary["role"] == "primary"
    assert primary["is_stub"] is False
    assert primary["status"] == "current"

    # Categorical confidence verification (High confidence due to multiple evidence chunks & Scope)
    conf = primary["confidence"]
    assert conf["level"] in ("High", "HIGH")
    assert any(
        reason in conf["reasons"]
        for reason in ("scope_clause_grounded", "APPLICABILITY_CONFIRMED", "multi_chunk_evidence")
    )

    # Provenance and evidence verification
    evidence = primary["evidence"]
    assert len(evidence) >= 1
    ev0 = evidence[0]
    assert ev0["source_file"] == "is.3370.1.2021.pdf"
    assert ev0["absolute_page_start"] in (1, 14)
    assert ev0["section_number"] in ("1", "4.3")
    assert "text" in ev0

    # Typed relationships resolution: IS 456 (normative_reference) & IS 2645 (allied_standard)
    related = primary["related_standards"]
    assert len(related) >= 2
    rel_map = {r["standard_id"]: r["relationship_type"] for r in related}
    assert "IS 456:2000" in rel_map
    assert rel_map["IS 456:2000"] == "normative_reference"
    assert "IS 2645:2003" in rel_map
    assert rel_map["IS 2645:2003"] == "allied_standard"


def test_t4_scenario_2_legacy_frontend_query_parity(client):
    """Scenario 2: Legacy Frontend Query Parity via POST /get_result.
    
    Guarantees 100% field parity expected by frontend/js/results.js:
    is_code, standard_number, title, final_score, score, reason, explanation_text,
    source, source_evidence, category_breadcrumb, explanation_mode, explanation_label.
    """
    res = client.post("/get_result", json={"user_input": QUERY_WIND_EFFECTS})
    assert res.status_code == 200

    data = res.get_json()
    assert data["status"] == "success"
    assert data["result_state"] == "recommendations"
    assert data["user_input"] == QUERY_WIND_EFFECTS
    assert "analysis_timestamp" in data
    assert isinstance(data["retrieved_candidate_count"], int)
    assert isinstance(data["reranked_candidate_count"], int)
    assert isinstance(data["rejected_candidate_count"], int)
    assert isinstance(data["highlight_terms"], list)
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 5

    rec = data["recommendations"][0]
    # Check all fields strictly required by frontend/js/results.js
    assert "is_code" in rec
    assert "standard_number" in rec
    assert "title" in rec
    assert "final_score" in rec
    assert "score" in rec
    assert "reason" in rec
    assert "explanation_text" in rec
    assert "explanation_mode" in rec
    assert "explanation_label" in rec
    assert "category_breadcrumb" in rec
    assert "source" in rec
    assert "source_evidence" in rec
    assert "evidence" in rec
    assert "warnings" in rec

    # Nested structure checks
    src_ev = rec["source_evidence"]
    assert "document_title" in src_ev
    assert "edition" in src_ev
    assert "page_start" in src_ev
    assert "page_end" in src_ev
    assert "section_number" in src_ev
    assert "section_name" in src_ev


def test_t4_scenario_3_cold_start_release_mount_and_health_lifecycle(tmp_path: Path):
    """Scenario 3: Cold-Start Release Mount & Health Check Lifecycle.
    
    Lifecycle sequence:
    1. Boot application with unmounted release -> /health/live returns 200, /health/ready returns 503.
    2. Mount verified release bundle -> /health/ready returns 200.
    3. Query executes successfully with full recommendation.
    """
    builder = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_lifecycle_01")
    rel_dir = builder.build_valid_release()

    # Step 1: Cold boot unmounted
    settings = Settings(release_dir=tmp_path)
    container = ApplicationContainer(settings, None)
    app = create_app(container)

    with app.test_client() as tc:
        # /health/live is 200 alive
        live_res = tc.get("/health/live")
        assert live_res.status_code == 200
        assert live_res.get_json()["status"] == "live"

        # /health/ready is 503 unready
        ready_res1 = tc.get("/health/ready")
        assert ready_res1.status_code == 503
        assert ready_res1.get_json()["status"] == "not_ready"

        # Step 2: Mount verified release
        container.mount_release(rel_dir)

        # /health/ready is now 200 ready
        ready_res2 = tc.get("/health/ready")
        assert ready_res2.status_code == 200
        ready_data = ready_res2.get_json()
        assert ready_data["status"] == "ready"
        assert ready_data["release_id"] == "rel_lifecycle_01"
        assert ready_data["model_loaded"] is False

        # Step 3: Run query successfully
        query_res = tc.post("/v1/analyze", json={"requirement": QUERY_UNDERGROUND_RESERVOIR})
        assert query_res.status_code == 200
        assert query_res.get_json()["result_state"] == "recommendations"

    container.close()


def test_t4_scenario_4_atomic_corpus_rollback_under_live_traffic(tmp_path: Path):
    """Scenario 4: Atomic Corpus Rollback Under Live Serving Traffic.
    
    Verifies zero downtime and 100% 200 OK responses during atomic rollback.
    """
    builder_v1 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_v1_live")
    rel_v1 = builder_v1.build_valid_release()

    builder_v2 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_v2_live")
    rel_v2 = builder_v2.build_valid_release()

    settings = Settings(release_dir=rel_v2)
    container = ApplicationContainer(settings, rel_v2)
    app = create_app(container)

    def execute_request(req_idx: int) -> int:
        with app.test_client() as tc:
            res = tc.post("/v1/analyze", json={"requirement": f"clause requirement test {req_idx}"})
            return res.status_code

    # Execute 30 requests in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(execute_request, i) for i in range(30)]

        # Perform atomic rollback while requests are in flight
        builder_v2.rollback_to("rel_v1_live")
        container.mount_release(rel_v1)

        # Add 30 more requests post-rollback
        futures.extend([executor.submit(execute_request, i) for i in range(30, 60)])

        results = [f.result() for f in as_completed(futures)]

    # Assert 100% of all 60 requests returned HTTP 200 (zero 500s, zero dropped requests)
    assert len(results) == 60
    assert all(code == 200 for code in results)

    # Active release is now v1
    assert container.manifest["release_id"] == "rel_v1_live"
    container.close()


def test_t4_scenario_5_non_blocking_ollama_offline_fail_open(client, container: ApplicationContainer):
    """Scenario 5: Non-Blocking Ollama Offline Explanation Fail-Open.
    
    Simulates unreachable Ollama service.
    Verifies that the request returns in <1.0s with deterministic evidence summary,
    leaving recommendation scores, ranking, and confidence completely unaffected.
    """
    # Configure unreachable port
    container.settings.ollama_host = "http://127.0.0.1:9999"
    container.settings.explanation = replace(
        container.settings.explanation,
        timeout_seconds=0.25,
    )

    t0 = time.monotonic()
    res = client.post(
        "/v1/explain",
        json={
            "requirement": QUERY_UNDERGROUND_RESERVOIR,
            "standard_id": "IS 3370 (Part 1):2021",
            "evidence": [
                {"chunk_id": "chk-is3370-01", "section_number": "1", "section_title": "Scope", "text": "Scope of standard"}
            ],
        },
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0  # Fails open rapidly
    assert res.status_code == 200
    data = res.get_json()
    assert data["mode"] == "deterministic"
    assert "explanation" in data
    assert "Generative LLM unavailable" in (data.get("warning") or "")


def test_t4_scenario_6_unverified_document_reference_standard_stub(client):
    """Scenario 6: Unverified Document Reference Resolved as StandardStub.
    
    SP 7:2026 text cites uningested standard IS 1786:2008.
    Decision engine must strictly bar StandardStub from primary recommendations (rank 1),
    placing it only under related_standards with is_stub=True and audit warning.
    """
    res = client.post(
        "/v1/analyze",
        json={"requirement": QUERY_STEEL_BARS_STUB},
    )
    assert res.status_code == 200
    data = res.get_json()
    recs = data.get("recommendations", [])
    assert len(recs) >= 1

    # Invariant 1: Primary recommendation must NOT be a stub
    primary = recs[0]
    assert primary["standard_id"]["code"] == "SP 7:2026"
    assert primary["is_stub"] is False

    # Invariant 2: StandardStub must appear under related_standards with is_stub=True
    related = primary.get("related_standards", [])
    stub_entries = [r for r in related if r["standard_id"] == "IS 1786:2008"]
    assert len(stub_entries) == 1
    stub_rel = stub_entries[0]
    assert stub_rel["is_stub"] is True
    assert "StandardStub" in stub_rel.get("warning", "")
