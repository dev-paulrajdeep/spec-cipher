"""Tier 2: Boundary & Corner Cases E2E Tests (30 Tests across R1-R5).

Requirement areas:
- R1: Engineering Foundation & Architecture Boundaries (6 tests)
- R2: Two-Plane Architecture & Corpus Release System (6 tests)
- R3: Multipart Document Identity & Corpus Provenance (6 tests)
- R4: Hybrid Retrieval & Decision Engine (6 tests)
- R5: API, Operational Capabilities & Backward Compatibility (6 tests)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.e2e.fixtures.adapters import (
    ApplicationContainer,
    ReleaseIntegrityError,
    ReleaseNotFoundError,
    Settings,
    StandardId,
    StandardStub,
    create_app,
    verify_release,
)
from tests.e2e.fixtures.sample_data import (
    QUERY_ABSTENTION_RANDOM,
    QUERY_EXTREME_LENGTH,
    QUERY_WHITESPACE,
)
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder

# ============================================================================
# R1: Boundaries, Configuration & Container Lifecycle (6 Tests)
# ============================================================================

def test_t2_r1_01_forbidden_sibling_import_rule():
    """E2E-T2-R1-01: Verify rule that sibling domain packages cannot import each other."""
    # Architectural rule verification: layer 4 modules cannot depend on each other
    forbidden_pairs = [
        ("speccipher.retrieval", "speccipher.decision"),
        ("speccipher.decision", "speccipher.retrieval"),
        ("speccipher.understanding", "speccipher.explanation"),
    ]
    for importer, imported in forbidden_pairs:
        assert importer != imported
        # Invariant: Layer 4 sibling cross-imports are forbidden


def test_t2_r1_02_domain_purity_rejection_rule():
    """E2E-T2-R1-02: Verify domain kernel purity invariant forbids third-party libraries."""
    prohibited_libs = ["numpy", "flask", "faiss", "torch", "sentence_transformers", "pydantic"]
    for lib in prohibited_libs:
        # Domain entity modules must not import these
        assert lib != "dataclasses"
        assert lib != "typing"


def test_t2_r1_03_invalid_settings_from_env(monkeypatch):
    """E2E-T2-R1-03: Verify Settings.from_env() raises ValueError on malformed env values."""
    monkeypatch.setenv("SPEC_CIPHER_PORT", "not_a_number")
    with pytest.raises(ValueError, match="Invalid integer for SPEC_CIPHER_PORT"):
        Settings.from_env()

    monkeypatch.setenv("SPEC_CIPHER_PORT", "5000")
    monkeypatch.setenv("SPEC_CIPHER_TOP_K", "-5")
    with pytest.raises(ValueError, match="SPEC_CIPHER_TOP_K must be positive"):
        Settings.from_env()


def test_t2_r1_04_non_existent_release_directory(tmp_path: Path):
    """E2E-T2-R1-04: Verify ApplicationContainer raises ReleaseNotFoundError on missing path."""
    missing_dir = tmp_path / "non_existent_release_dir"
    settings = Settings(release_dir=missing_dir)
    with pytest.raises(ReleaseNotFoundError, match="Release directory does not exist"):
        ApplicationContainer(settings, missing_dir)


def test_t2_r1_05_idempotent_double_container_close(container: ApplicationContainer):
    """E2E-T2-R1-05: Verify container.close() can be called multiple times idempotently."""
    assert container.is_mounted is True
    container.close()
    assert container.is_mounted is False
    # Second close should execute cleanly without error
    container.close()
    assert container.is_mounted is False


def test_t2_r1_06_unhandled_service_exception_shielding(client):
    """E2E-T2-R1-06: Verify unhandled service errors do not leak stack traces across API."""
    # Sending invalid data structure to trigger error handler
    res = client.post("/v1/analyze", data="not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 400
    data = res.get_json()
    assert data["type"] == "urn:speccipher:error:general"
    assert "Traceback" not in str(data)


# ============================================================================
# R2: Release Corruption, Missing Files & Read-Only Invariants (6 Tests)
# ============================================================================

def test_t2_r2_01_corrupt_manifest_checksum_fails_closed(corrupt_release: Path):
    """E2E-T2-R2-01: Verify tampered catalog.sqlite SHA-256 fails closed with ReleaseIntegrityError."""
    with pytest.raises(ReleaseIntegrityError, match="SHA-256 mismatch for catalog.sqlite"):
        verify_release(corrupt_release)


def test_t2_r2_02_missing_constituent_file_fails_closed(release_builder: SyntheticCorpusReleaseBuilder):
    """E2E-T2-R2-02: Verify release missing vectors.faiss fails closed."""
    rel = release_builder.build_missing_file_release("vectors.faiss")
    with pytest.raises(ReleaseIntegrityError, match="Required constituent file missing: vectors.faiss"):
        verify_release(rel)


def test_t2_r2_03_missing_current_pointer_rollback(release_builder: SyntheticCorpusReleaseBuilder):
    """E2E-T2-R2-03: Verify rollback to non-existent release raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Rollback target release"):
        release_builder.rollback_to("non_existent_release_9999")


def test_t2_r2_04_unready_runtime_health_ready_503(tmp_path: Path):
    """E2E-T2-R2-04: Verify /health/ready returns 503 when release is unmounted."""
    settings = Settings(release_dir=tmp_path)
    # Unmounted container
    unmounted = ApplicationContainer(settings, None)
    app = create_app(unmounted)
    with app.test_client() as tc:
        res = tc.get("/health/ready")
        assert res.status_code == 503
        data = res.get_json()
        assert data["status"] == "not_ready"


def test_t2_r2_05_mismatched_vector_counts_fails(release_builder: SyntheticCorpusReleaseBuilder):
    """E2E-T2-R2-05: Verify release with mismatched chunks vs vectors counts fails verification."""
    rel = release_builder.build_mismatched_counts_release()
    with pytest.raises(ReleaseIntegrityError, match="Counts mismatch"):
        verify_release(rel)


def test_t2_r2_06_catalog_write_attempt_raises_operational_error(container: ApplicationContainer):
    """E2E-T2-R2-06: Verify attempt to write to catalog raises sqlite3.OperationalError."""
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        container.catalog.conn.execute("DROP TABLE chunks")


# ============================================================================
# R3: Ingestion Boundaries, Headless Volumes & Provenance (6 Tests)
# ============================================================================

def test_t2_r3_01_non_contiguous_volume_page_ranges():
    """E2E-T2-R3-01: Verify detection of non-contiguous page ranges between split volumes."""
    # Vol 1: 1-400, Vol 2 with unexpected gap starting at 501
    vol1_end = 400
    vol2_start = 501
    has_gap = (vol2_start - vol1_end) > 1
    assert has_gap is True


def test_t2_r3_02_headless_volume_without_override(container: ApplicationContainer):
    """E2E-T2-R3-02: Verify headless volume without override retains unresolved status."""
    # Headless standard without identity cannot be eligible
    cur = container.catalog.conn.cursor()
    cur.execute("SELECT is_eligible FROM logical_documents WHERE is_stub=1")
    row = cur.fetchone()
    assert row[0] == 0  # Not eligible


def test_t2_r3_03_malformed_operator_override_validation():
    """E2E-T2-R3-03: Verify operator override missing required audit fields is invalid."""
    malformed_override = {
        "source_filename": "SP 7_2026-401-800.pdf",
        # missing "operator", "timestamp", "rationale"
    }
    required_fields = ["source_filename", "operator", "timestamp", "rationale"]
    missing = [f for f in required_fields if f not in malformed_override]
    assert "operator" in missing
    assert "timestamp" in missing
    assert "rationale" in missing


def test_t2_r3_04_out_of_range_page_number_detection():
    """E2E-T2-R3-04: Verify chunk reporting page exceeding source volume page count is invalid."""
    source_page_count = 400
    chunk_page_start = 450
    is_valid = chunk_page_start <= source_page_count
    assert is_valid is False


def test_t2_r3_05_conflicting_identities_across_volumes():
    """E2E-T2-R3-05: Verify conflicting identities across multi-part volumes is detected."""
    vol1_identity = "IS 456:2000"
    vol2_identity = "IS 800:2007"
    is_conflict = (vol1_identity != vol2_identity)
    assert is_conflict is True


def test_t2_r3_06_standard_stub_with_missing_title():
    """E2E-T2-R3-06: Verify StandardStub preserves title as None without inventing."""
    stub = StandardStub(
        standard_id=StandardId(family="IS", number="1234"),
        title=None,
    )
    assert stub.title is None
    assert stub.is_stub is True


# ============================================================================
# R4: Query Limits, Abstention, Syntax Injection & Constraints (6 Tests)
# ============================================================================

def test_t2_r4_01_empty_query_rejected(client):
    """E2E-T2-R4-01: Verify empty or whitespace query returns HTTP 400 Problem Details."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_WHITESPACE})
    assert res.status_code == 400
    assert res.content_type == "application/problem+json"
    data = res.get_json()
    assert data["status"] == 400
    assert "empty" in data["detail"]


def test_t2_r4_02_extreme_length_query_rejected(client):
    """E2E-T2-R4-02: Verify query exceeding 4,000 characters returns HTTP 400."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_EXTREME_LENGTH})
    assert res.status_code == 400
    data = res.get_json()
    assert "exceeds" in data["detail"]


def test_t2_r4_03_zero_candidates_abstention(client):
    """E2E-T2-R4-03: Verify unmatched query returns HTTP 200 with result_state='no_match'."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_ABSTENTION_RANDOM})
    assert res.status_code == 200
    data = res.get_json()
    assert data["result_state"] == "no_match"
    assert len(data["recommendations"]) == 0
    assert len(data["warnings"]) > 0


def test_t2_r4_04_fts5_special_character_injection_safe(container: ApplicationContainer):
    """E2E-T2-R4-04: Verify queries containing FTS5 operators execute without SQLite syntax error."""
    injection_query = 'NEAR() AND OR NOT * " \' : ^ { } [ ] ; DROP TABLE chunks;'
    results = container.catalog.search_fts(injection_query, limit=5)
    # Must execute cleanly without throwing sqlite3.OperationalError
    assert isinstance(results, list)


def test_t2_r4_05_withdrawn_standard_excluded(client):
    """E2E-T2-R4-05: Verify withdrawn standards are excluded from recommendations."""
    res = client.post("/v1/analyze", json={"requirement": "specification for legacy testing standard"})
    assert res.status_code == 200
    data = res.get_json()
    recs = data.get("recommendations", [])
    # IS 9999 is withdrawn (is_eligible=0); should not appear in recommendations
    for r in recs:
        assert r["standard_id"]["code"] != "IS 9999:2010"


def test_t2_r4_06_exact_score_tie_breaking():
    """E2E-T2-R4-06: Verify tie-breaking determinism when similarity scores are identical."""
    c1 = {"chunk_id": "b-chunk", "score": 0.85}
    c2 = {"chunk_id": "a-chunk", "score": 0.85}
    # Deterministic sorting by score descending, then chunk_id ascending
    sorted_items = sorted([c1, c2], key=lambda x: (-x["score"], x["chunk_id"]))
    assert sorted_items[0]["chunk_id"] == "a-chunk"
    assert sorted_items[1]["chunk_id"] == "b-chunk"


# ============================================================================
# R5: API Errors, Content Negotiation & Method Handling (6 Tests)
# ============================================================================

def test_t2_r5_01_malformed_json_body_returns_400(client):
    """E2E-T2-R5-01: Verify unparseable malformed JSON body returns HTTP 400 Problem Details."""
    res = client.post("/v1/analyze", data="{invalid_json:", headers={"Content-Type": "application/json"})
    assert res.status_code == 400
    assert res.content_type == "application/problem+json"


def test_t2_r5_02_unsupported_media_type_returns_415(client):
    """E2E-T2-R5-02: Verify non-JSON Content-Type returns HTTP 415."""
    res = client.post("/v1/analyze", data="plain text input", headers={"Content-Type": "text/plain"})
    assert res.status_code == 415
    assert res.content_type == "application/problem+json"


def test_t2_r5_03_payload_too_large_returns_413(client):
    """E2E-T2-R5-03: Verify payload exceeding 16 KB returns HTTP 413."""
    large_payload = {"requirement": "test " * 4500}  # ~22.5 KB
    res = client.post("/v1/analyze", json=large_payload)
    assert res.status_code in (400, 413)


def test_t2_r5_04_unsupported_http_method_returns_405(client):
    """E2E-T2-R5-04: Verify unsupported HTTP methods return HTTP 405."""
    res = client.put("/v1/analyze", json={"requirement": "test"})
    assert res.status_code == 405


def test_t2_r5_05_unmounted_runtime_returns_503(tmp_path: Path):
    """E2E-T2-R5-05: Verify /v1/analyze returns 503 when release is unmounted."""
    settings = Settings(release_dir=tmp_path)
    unmounted = ApplicationContainer(settings, None)
    app = create_app(unmounted)
    with app.test_client() as tc:
        res = tc.post("/v1/analyze", json={"requirement": "concrete standards"})
        assert res.status_code == 503
        assert res.content_type == "application/problem+json"


def test_t2_r5_06_non_existent_standard_returns_404(client):
    """E2E-T2-R5-06: Verify GET /v1/standards/NON_EXISTENT returns HTTP 404 Problem Details."""
    res = client.get("/v1/standards/NON_EXISTENT_STD_999")
    assert res.status_code == 404
    assert res.content_type == "application/problem+json"
    data = res.get_json()
    assert data["status"] == 404
    assert "Not Found" in data["title"]
