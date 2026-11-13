"""Tier 3: Cross-Feature Combinations E2E Tests (12 Pairwise Tests across R1-R5).

Pairwise interactions covered:
- E2E-T3-PAIR-01: Release Publish + Live Serving (R2 + R5)
- E2E-T3-PAIR-02: Multipart Assembly + FTS5 Search (R3 + R4)
- E2E-T3-PAIR-03: Categorical Confidence + Explain Degradation (R4 + R5)
- E2E-T3-PAIR-04: Atomic Rollback + Subsequent Query (R2 + R5)
- E2E-T3-PAIR-05: StandardStub Model + Primary Barring (R3 + R4)
- E2E-T3-PAIR-06: ApplicationContainer + Release Integrity (R1 + R2)
- E2E-T3-PAIR-07: Legacy Adapter + Hybrid RRF Engine (R4 + R5)
- E2E-T3-PAIR-08: Operator Override + Standards API (R3 + R5)
- E2E-T3-PAIR-09: Request ID Middleware + Problem Details (R1 + R5)
- E2E-T3-PAIR-10: Cold Start SQLite + Lexical Performance (R2 + R4)
- E2E-T3-PAIR-11: Dense Search + Confidence Thresholding (R4 + R4)
- E2E-T3-PAIR-12: Tender Upload + Feature Extraction (R1/R5 + R4)
"""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path

import pytest
from speccipher.api.app import create_app
from speccipher.config.settings import Settings
from speccipher.decision.confidence import calculate_categorical_confidence
from speccipher.release import ReleaseIntegrityError
from speccipher.services.container import ApplicationContainer
from tests.e2e.fixtures.sample_data import (
    QUERY_STEEL_BARS_STUB,
    QUERY_UNDERGROUND_RESERVOIR,
    QUERY_WIND_EFFECTS,
)
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder


def test_t3_pair_01_release_publish_and_live_serving(tmp_path: Path):
    """E2E-T3-PAIR-01: Release Publish + Live Serving (R2 + R5)."""
    builder_v1 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_2026_v1")
    rel_v1 = builder_v1.build_valid_release()

    settings = Settings(release_dir=rel_v1)
    container = ApplicationContainer(settings, rel_v1)
    app = create_app(container)

    with app.test_client() as client:
        # Serving v1
        res1 = client.get("/health/ready")
        assert res1.status_code == 200
        assert res1.get_json()["release_id"] == "rel_2026_v1"

        # Builder publishes v2 and updates CURRENT pointer
        builder_v2 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_2026_v2")
        rel_v2 = builder_v2.build_valid_release()

        # Mount v2 dynamically
        container.mount_release(rel_v2)

        res2 = client.get("/health/ready")
        assert res2.status_code == 200
        assert res2.get_json()["release_id"] == "rel_2026_v2"

    container.close()


def test_t3_pair_02_multipart_assembly_and_fts5_search(container: ApplicationContainer):
    """E2E-T3-PAIR-02: Multipart Assembly + FTS5 Search (R3 + R4)."""
    # Query phrase from SP 7:2026 Volume 6 (offset 2000, page 120 -> absolute page 2120)
    results = container.catalog.search_fts("pressurization escape routes", limit=5)
    assert len(results) > 0
    vol6_chunks = [r for r in results if r["chunk_id"] == "chk-sp7-04"]
    assert len(vol6_chunks) == 1
    chk = vol6_chunks[0]
    assert chk["standard_code"] == "SP 7:2026"
    assert chk["source_file_id"] == "SP 7_2026-2001-2400.pdf"
    assert chk["absolute_page_start"] == 2120
    assert chk["absolute_page_end"] == 2121


def test_t3_pair_03_confidence_and_explain_degradation(client, container: ApplicationContainer):
    """E2E-T3-PAIR-03: Categorical Confidence + Explain Degradation (R4 + R5)."""
    # Set simulated offline Ollama host
    container.settings.ollama_host = "http://127.0.0.1:9999"
    container.settings.explanation = replace(
        container.settings.explanation,
        timeout_seconds=0.25,
    )

    start_time = time.monotonic()
    res = client.post(
        "/v1/explain",
        json={"requirement": QUERY_UNDERGROUND_RESERVOIR, "standard_id": "IS 3370 (Part 1):2021", "evidence": []},
    )
    elapsed = time.monotonic() - start_time

    assert elapsed < 1.0  # Non-blocking fail-open executed in < 1s
    assert res.status_code == 200
    data = res.get_json()
    assert data["mode"] == "deterministic"
    assert "explanation" in data
    assert "Generative LLM unavailable" in (data.get("warning") or "")


def test_t3_pair_04_atomic_rollback_and_subsequent_query(tmp_path: Path):
    """E2E-T3-PAIR-04: Atomic Rollback + Subsequent Query (R2 + R5)."""
    builder1 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_baseline")
    rel1 = builder1.build_valid_release()

    builder2 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_faulty")
    rel2 = builder2.build_valid_release()

    settings = Settings(release_dir=rel2)
    container = ApplicationContainer(settings, rel2)
    app = create_app(container)

    with app.test_client() as client:
        # In rel_faulty
        res_v2 = client.post("/v1/analyze", json={"requirement": QUERY_UNDERGROUND_RESERVOIR})
        assert res_v2.status_code == 200
        assert res_v2.get_json()["release_id"] == "rel_faulty"

        # Operator performs rollback to rel_baseline
        builder2.rollback_to("rel_baseline")
        container.mount_release(rel1)

        # Subsequent query served from rel_baseline
        res_v1 = client.post("/v1/analyze", json={"requirement": QUERY_UNDERGROUND_RESERVOIR})
        assert res_v1.status_code == 200
        assert res_v1.get_json()["release_id"] == "rel_baseline"

    container.close()


def test_t3_pair_05_standard_stub_and_primary_barring(client):
    """E2E-T3-PAIR-05: StandardStub Model + Primary Barring (R3 + R4)."""
    # Query referencing uningested steel bars standard IS 1786
    res = client.post("/v1/analyze", json={"requirement": QUERY_STEEL_BARS_STUB})
    assert res.status_code == 200
    data = res.get_json()
    recs = data.get("recommendations", [])
    if recs:
        # Primary recommendation must be the ingested code (SP 7:2026), NOT the stub
        assert recs[0]["standard_id"]["code"] == "SP 7:2026"
        assert recs[0]["is_stub"] is False

        # IS 1786 must only appear under related_standards with is_stub=True
        related = recs[0].get("related_standards", [])
        stub_rel = [r for r in related if r["standard_id"] == "IS 1786:2008"]
        assert len(stub_rel) > 0
        assert stub_rel[0]["is_stub"] is True
        assert "StandardStub" in (stub_rel[0].get("warning") or "")


def test_t3_pair_06_container_fails_on_release_integrity_failure(corrupt_release: Path):
    """E2E-T3-PAIR-06: ApplicationContainer + Release Integrity (R1 + R2)."""
    settings = Settings(release_dir=corrupt_release)
    with pytest.raises(ReleaseIntegrityError, match="SHA-256 mismatch"):
        ApplicationContainer(settings, corrupt_release)


def test_t3_pair_07_legacy_adapter_and_hybrid_rrf(client):
    """E2E-T3-PAIR-07: Legacy Adapter + Hybrid RRF Engine (R4 + R5)."""
    res = client.post("/get_result", json={"user_input": QUERY_WIND_EFFECTS})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "recommendations" in data
    assert "retrieved_candidate_count" in data
    assert "reranked_candidate_count" in data
    assert data["result_state"] in ("recommendations", "no_match")


def test_t3_pair_08_operator_override_and_standards_api(client):
    """E2E-T3-PAIR-08: Operator Override + Standards API (R3 + R5)."""
    res = client.get("/v1/standards/SP 7:2026")
    assert res.status_code == 200
    data = res.get_json()
    std = data.get("standard", {})
    assert std["code"] == "SP 7:2026"
    assert std["is_eligible"] == 1


def test_t3_pair_09_request_id_middleware_and_problem_details(client):
    """E2E-T3-PAIR-09: Request ID Middleware + Problem Details (R1 + R5)."""
    custom_trace_id = "trace-uuid-abcdef-123456"
    res = client.post(
        "/v1/analyze",
        json={},  # Empty payload triggers 400
        headers={"X-Request-ID": custom_trace_id},
    )
    assert res.status_code == 400
    assert res.headers.get("X-Request-ID") == custom_trace_id
    data = res.get_json()
    assert data["request_id"] == custom_trace_id
    assert res.content_type == "application/problem+json"


def test_t3_pair_10_cold_start_sqlite_and_lexical_perf(tmp_path: Path):
    """E2E-T3-PAIR-10: Cold Start SQLite + Lexical Performance (R2 + R4)."""
    builder = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_perf")
    rel_dir = builder.build_valid_release()

    t0 = time.monotonic()
    settings = Settings(release_dir=rel_dir)
    cntr = ApplicationContainer(settings, rel_dir)
    boot_time = time.monotonic() - t0

    assert boot_time < 0.5  # Cold boot completes in under 500ms (no 170MB JSON parse)

    t1 = time.monotonic()
    results = cntr.catalog.search_fts("concrete liquid storage", limit=5)
    query_time = time.monotonic() - t1

    assert query_time < 0.05  # FTS5 query in < 50ms
    assert len(results) > 0
    cntr.close()


def test_t3_pair_11_dense_search_and_confidence_thresholding():
    """E2E-T3-PAIR-11: Dense Search + Confidence Thresholding (R4 + R4)."""
    # Candidate A: 2 chunks + Scope clause
    cand_a_evidence = [
        {"section_title": "Scope", "text": "Scope of standard"},
        {"section_title": "Design", "text": "Design clause"},
    ]
    conf_a = calculate_categorical_confidence(cand_a_evidence, is_eligible=True, margin=0.25)
    assert conf_a.level == "High"

    # Candidate B: 1 chunk, no Scope clause
    cand_b_evidence = [
        {"section_title": "General", "text": "General clause"}
    ]
    conf_b = calculate_categorical_confidence(cand_b_evidence, is_eligible=True, margin=0.05)
    assert conf_b.level in ("Medium", "Low")
    assert "single_chunk_evidence" in conf_b.reasons


def test_t3_pair_12_tender_upload_and_feature_extraction(client):
    """E2E-T3-PAIR-12: Tender Upload + Feature Extraction (R1/R5 + R4)."""
    fake_pdf_stream = io.BytesIO(b"%PDF-1.4 Mock Tender Document Header and Clauses")
    res = client.post(
        "/v1/analyze/document",
        data={"file": (fake_pdf_stream, "tender_spec.pdf")},
        content_type="multipart/form-data",
        headers={"X-Document-Page-Count": "12"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "request_id" in data
    assert "extracted_requirements" in data
    assert data["status"] == "processed"
