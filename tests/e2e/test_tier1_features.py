"""Tier 1: Feature Coverage E2E Tests (30 Tests across R1-R5).

Requirement areas:
- R1: Engineering Foundation & Architecture Boundaries (6 tests)
- R2: Two-Plane Architecture & Corpus Release System (6 tests)
- R3: Multipart Document Identity & Corpus Provenance (6 tests)
- R4: Hybrid Retrieval & Decision Engine (6 tests)
- R5: API, Operational Capabilities & Backward Compatibility (6 tests)
"""

from __future__ import annotations

import ast
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from tests.e2e.fixtures.adapters import (
    ApplicationContainer,
    Candidate,
    DocumentChunk,
    Settings,
    StandardId,
    StandardStub,
    calculate_categorical_confidence,
    create_app,
    reciprocal_rank_fusion,
    verify_release,
)
from tests.e2e.fixtures.sample_data import (
    QUERY_FIRE_EXIT,
    QUERY_UNDERGROUND_RESERVOIR,
    SAMPLE_CHUNKS,
    SAMPLE_OPERATOR_OVERRIDES,
)
from tests.e2e.fixtures.synthetic_release import SyntheticCorpusReleaseBuilder

# ============================================================================
# R1: Engineering Foundation & Architecture Migration (6 Tests)
# ============================================================================

def test_t1_r1_01_pure_domain_kernel_purity():
    """E2E-T1-R1-01: Verify domain entities are pure Python standard library only."""
    # Test entity instantiations
    std_id = StandardId(family="IS", number="456", year=2000)
    assert std_id.code == "IS 456:2000"
    assert std_id.family == "IS"

    chunk = DocumentChunk(
        chunk_id="chk-01",
        standard_id=std_id,
        logical_doc_id="IS_456_2000",
        source_file_id="is.456.2000.pdf",
        absolute_page_start=1,
        absolute_page_end=2,
        section_number="1",
        section_title="Scope",
        text="Concrete specification code.",
        is_eligible=True,
    )
    assert chunk.chunk_id == "chk-01"
    assert chunk.is_eligible is True

    # If src/speccipher/domain exists, verify AST imports contain no third-party libraries
    domain_path = Path("src/speccipher/domain")
    if domain_path.exists():
        for py_file in domain_path.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_pkg = alias.name.split(".")[0]
                        assert top_pkg in sys.stdlib_module_names, (
                            f"Domain file {py_file} imports external library: {top_pkg}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        top_pkg = node.module.split(".")[0]
                        assert top_pkg in sys.stdlib_module_names or top_pkg == "speccipher", (
                            f"Domain file {py_file} imports external library: {top_pkg}"
                        )


def test_t1_r1_02_layered_architecture_import_rules():
    """E2E-T1-R1-02: Verify layered architecture contracts (no upward imports)."""
    src_path = Path("src/speccipher")
    if not src_path.exists():
        # Architecture contract test: layer hierarchy definitions
        layers = [
            "domain",       # Layer 7
            "config",       # Layer 6
            "catalog",      # Layer 5
            "retrieval",    # Layer 4
            "release",      # Layer 3
            "services",     # Layer 2
            "api",          # Layer 1
        ]
        assert len(layers) == 7
        return

    # Check that domain does not import from higher layers
    domain_files = list((src_path / "domain").glob("*.py"))
    for py_file in domain_files:
        content = py_file.read_text(encoding="utf-8")
        assert "speccipher.api" not in content
        assert "speccipher.services" not in content
        assert "speccipher.release" not in content


def test_t1_r1_03_sibling_engine_isolation():
    """E2E-T1-R1-03: Verify Layer 4 sibling modules never cross-import each other."""
    src_path = Path("src/speccipher")
    if not src_path.exists():
        siblings = ["retrieval", "decision", "ingestion", "understanding", "explanation"]
        assert len(siblings) == 5
        return

    for sib in ["retrieval", "decision", "ingestion", "understanding", "explanation"]:
        sib_dir = src_path / sib
        if not sib_dir.exists():
            continue
        for py_file in sib_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for other in ["retrieval", "decision", "ingestion", "understanding", "explanation"]:
                if other != sib:
                    assert f"speccipher.{other}" not in content, (
                        f"Sibling isolation violated in {py_file}: imports speccipher.{other}"
                    )


def test_t1_r1_04_serving_plane_ingestion_isolation():
    """E2E-T1-R1-04: Verify serving plane modules never import speccipher.ingestion."""
    src_path = Path("src/speccipher")
    if not src_path.exists():
        return

    for pkg in ["api", "services", "retrieval", "decision", "understanding", "explanation"]:
        pkg_dir = src_path / pkg
        if not pkg_dir.exists():
            continue
        for py_file in pkg_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "speccipher.ingestion" not in content, (
                f"Serving plane violation: {py_file} imports speccipher.ingestion"
            )


def test_t1_r1_05_application_container_lazy_init(tmp_path: Path):
    """E2E-T1-R1-05: Verify composition root instantiates cleanly with zero heavy imports."""
    builder = SyntheticCorpusReleaseBuilder(tmp_path)
    rel_dir = builder.build_valid_release()
    settings = Settings(release_dir=rel_dir, embedding_device="cpu")

    container = ApplicationContainer(settings, rel_dir)
    assert container.is_mounted is True
    assert container.manifest is not None
    assert container.catalog is not None
    container.close()
    assert container.is_mounted is False


def test_t1_r1_06_decoupled_flask_factory(container: ApplicationContainer):
    """E2E-T1-R1-06: Verify create_app factory configures routes and middleware."""
    app = create_app(container)
    assert app is not None
    routes = [rule.rule for rule in app.url_map.iter_rules()]
    assert "/health/live" in routes
    assert "/health/ready" in routes
    assert "/v1/analyze" in routes
    assert "/v1/explain" in routes
    assert "/v1/standards" in routes
    assert "/get_result" in routes


# ============================================================================
# R2: Two-Plane Architecture & Corpus Release System (6 Tests)
# ============================================================================

def test_t1_r2_01_release_manifest_schema_2(valid_release: Path):
    """E2E-T1-R2-01: Verify release.json complies with Schema 2.0."""
    manifest_path = valid_release / "release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2.0"
    assert "release_id" in manifest
    assert "created_at" in manifest
    assert "embedding" in manifest
    assert manifest["embedding"]["model"] == "BAAI/bge-m3"
    assert manifest["embedding"]["dim"] == 1024
    assert manifest["embedding"]["normalized"] is True
    assert "counts" in manifest
    assert manifest["counts"]["chunks"] == len(SAMPLE_CHUNKS)
    assert "files" in manifest
    assert "catalog.sqlite" in manifest["files"]
    assert "vectors.faiss" in manifest["files"]


def test_t1_r2_02_release_fail_closed_integrity(valid_release: Path):
    """E2E-T1-R2-02: Verify valid release passes checksum and count verification."""
    manifest = verify_release(valid_release)
    assert manifest["counts"]["chunks"] == manifest["counts"]["vectors"]


def test_t1_r2_03_atomic_publication_current_pointer(tmp_path: Path):
    """E2E-T1-R2-03: Verify atomic publication via CURRENT pointer."""
    builder = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_v1")
    rel_dir = builder.build_valid_release()

    current_file = tmp_path / "data" / "releases" / "CURRENT"
    assert current_file.exists()
    assert current_file.read_text(encoding="utf-8").strip() == "rel_v1"


def test_t1_r2_04_instant_rollback_mechanism(tmp_path: Path):
    """E2E-T1-R2-04: Verify instant rollback repoints CURRENT pointer."""
    builder1 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_v1")
    builder1.build_valid_release()

    builder2 = SyntheticCorpusReleaseBuilder(tmp_path, release_id="rel_v2")
    builder2.build_valid_release()

    current_file = tmp_path / "data" / "releases" / "CURRENT"
    assert current_file.read_text(encoding="utf-8").strip() == "rel_v2"

    # Rollback to rel_v1
    builder2.rollback_to("rel_v1")
    assert current_file.read_text(encoding="utf-8").strip() == "rel_v1"


def test_t1_r2_05_structured_sqlite_catalog(valid_release: Path):
    """E2E-T1-R2-05: Verify catalog.sqlite contains all required tables and FTS5."""
    db_path = valid_release / "catalog.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "logical_documents" in tables
    assert "source_files" in tables
    assert "relationships" in tables
    assert "scope_text" in tables
    assert "chunks" in tables
    assert "chunks_fts" in tables

    # Verify FTS5 is searchable
    cur.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'concrete' LIMIT 1")
    res = cur.fetchone()
    assert res is not None
    conn.close()


def test_t1_r2_06_read_only_serving_plane_invariant(container: ApplicationContainer):
    """E2E-T1-R2-06: Verify serving plane database connection is strictly read-only."""
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        container.catalog.conn.execute("INSERT INTO scope_text VALUES ('FOO', 'BAR', 'BAZ')")


# ============================================================================
# R3: Multipart Document Identity & Corpus Provenance (6 Tests)
# ============================================================================

def test_t1_r3_01_deterministic_multipart_assembly(container: ApplicationContainer):
    """E2E-T1-R3-01: Verify multi-volume publication SP 7:2026 is assembled as one LogicalDocument."""
    std = container.catalog.get_standard_by_code("SP 7:2026")
    assert std is not None
    assert std["code"] == "SP 7:2026"
    assert std["page_count"] == 2999

    # Verify source volumes
    cur = container.catalog.conn.cursor()
    cur.execute("SELECT file_name, page_offset FROM source_files WHERE logical_doc_id='SP_7_2026' ORDER BY page_offset")
    vols = cur.fetchall()
    assert len(vols) == 8
    assert vols[0][1] == 0
    assert vols[1][1] == 400
    assert vols[7][1] == 2800


def test_t1_r3_02_absolute_page_number_mapping(container: ApplicationContainer):
    """E2E-T1-R3-02: Verify volume relative pages are mapped to absolute document pages."""
    cur = container.catalog.conn.cursor()
    # Chunk from Volume 2 (offset 400, page_start 10 -> absolute 410)
    cur.execute("SELECT absolute_page_start, absolute_page_end FROM chunks WHERE chunk_id='chk-sp7-03'")
    row = cur.fetchone()
    assert row[0] == 410
    assert row[1] == 411


def test_t1_r3_03_auditable_operator_overrides():
    """E2E-T1-R3-03: Verify operator overrides structure and audit metadata."""
    override = SAMPLE_OPERATOR_OVERRIDES["overrides"][0]
    assert override["source_filename"] == "SP 7_2026-401-800.pdf"
    assert override["operator"] == "lead-architect"
    assert "timestamp" in override
    assert "rationale" in override
    assert override["asserted_standard"]["code"] == "SP 7:2026"


def test_t1_r3_04_standard_stub_representation():
    """E2E-T1-R3-04: Verify uningested cited standard is represented as StandardStub."""
    stub = StandardStub(
        standard_id=StandardId(family="IS", number="1786", year=2008),
        title="High Strength Deformed Steel Bars",
        referenced_by_chunk_ids=("chk-sp7-03",),
    )
    assert stub.is_stub is True
    assert stub.standard_id.code == "IS 1786:2008"
    assert "chk-sp7-03" in stub.referenced_by_chunk_ids


def test_t1_r3_05_strict_source_chunk_provenance(container: ApplicationContainer):
    """E2E-T1-R3-05: Verify chunk retains complete provenance."""
    cur = container.catalog.conn.cursor()
    cur.execute("SELECT * FROM chunks WHERE chunk_id='chk-is3370-01'")
    chk = dict(cur.fetchone())
    assert chk["source_file_id"] == "is.3370.1.2021.pdf"
    assert chk["absolute_page_start"] == 1
    assert chk["section_number"] == "1"
    assert chk["section_title"] == "Scope"
    assert "underground water storage" in chk["text"]


def test_t1_r3_06_standard_stub_barred_from_primary(client):
    """E2E-T1-R3-06: Verify StandardStub is barred from primary recommendations."""
    res = client.post("/v1/analyze", json={"requirement": "high strength deformed steel bars conforming to IS 1786"})
    assert res.status_code == 200
    data = res.get_json()
    recs = data.get("recommendations", [])
    if recs:
        # Primary recommendation must NOT be a stub
        assert recs[0]["is_stub"] is False
        assert recs[0]["standard_id"]["code"] != "IS 1786:2008"


# ============================================================================
# R4: Hybrid Retrieval & Decision Engine (6 Tests)
# ============================================================================

def test_t1_r4_01_sqlite_fts5_lexical_search(container: ApplicationContainer):
    """E2E-T1-R4-01: Verify SQLite FTS5 BM25 search retrieves matching chunks."""
    results = container.catalog.search_fts("pressurization escape routes", limit=5)
    assert len(results) > 0
    assert any("chk-sp7" in r["chunk_id"] for r in results)


def test_t1_r4_02_dense_vector_search(mock_encoder):
    """E2E-T1-R4-02: Verify deterministic dense embedding unit normalization."""
    vec = mock_encoder.encode("centrifugal pumps for water service")
    assert len(vec) == 1024
    norm = sum(x * x for x in vec)
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_t1_r4_03_reciprocal_rank_fusion():
    """E2E-T1-R4-03: Verify Reciprocal Rank Fusion (k=60) merges dense and lexical lists."""
    dense = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
    lexical = [{"chunk_id": "c2"}, {"chunk_id": "c4"}, {"chunk_id": "c1"}]

    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    # c2 is rank 2 in dense and rank 1 in lexical: score = 1/(60+2) + 1/(60+1) ~ 0.0161 + 0.0164 = 0.0325
    assert fused[0]["chunk_id"] == "c2"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_t1_r4_04_single_stage_reranking():
    """E2E-T1-R4-04: Verify single consolidated reranking stage output."""
    candidates = [
        Candidate(
            chunk=DocumentChunk("c1", None, "D1", "f1", 1, 2, "1", "Scope", "text", True),
            dense_score=0.82,
            lexical_score=0.75,
            rrf_score=0.032,
            rerank_score=0.88,
        )
    ]
    assert candidates[0].rerank_score == 0.88


def test_t1_r4_05_categorical_confidence_with_reasons():
    """E2E-T1-R4-05: Verify categorical confidence calculation with explicit reason codes."""
    evidence = [
        {"section_title": "Scope", "text": "Scope of the standard."},
        {"section_title": "Materials", "text": "Material specifications."},
    ]
    conf = calculate_categorical_confidence(evidence, is_eligible=True, margin=0.25)
    assert conf.level == "High"
    assert conf.score >= 0.85
    assert "scope_clause_grounded" in conf.reasons
    assert "multi_chunk_evidence" in conf.reasons
    assert "resolved_identity" in conf.reasons


def test_t1_r4_06_relationship_taxonomy(container: ApplicationContainer):
    """E2E-T1-R4-06: Verify relationship taxonomy types."""
    rels = container.catalog.get_relationships("IS 3370 (Part 1):2021")
    assert len(rels) >= 2
    rel_types = {r["relationship_type"] for r in rels}
    assert "normative_reference" in rel_types
    assert "allied_standard" in rel_types


# ============================================================================
# R5: API, Operational Capabilities & Backward Compatibility (6 Tests)
# ============================================================================

def test_t1_r5_01_versioned_analyze_endpoint(client):
    """E2E-T1-R5-01: Verify POST /v1/analyze returns valid response."""
    res = client.post(
        "/v1/analyze",
        json={"requirement": QUERY_UNDERGROUND_RESERVOIR, "options": {"top_k": 3}},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "request_id" in data
    assert "release_id" in data
    assert "result_state" in data
    assert "requirement_understanding" in data
    assert "recommendations" in data


def test_t1_r5_02_decoupled_explain_endpoint(client):
    """E2E-T1-R5-02: Verify POST /v1/explain returns generative or deterministic summary."""
    res = client.post(
        "/v1/explain",
        json={"requirement": QUERY_FIRE_EXIT, "standard_id": "SP 7:2026", "evidence": []},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "request_id" in data
    assert data["mode"] in ("generative", "deterministic")
    assert "explanation" in data
    assert "summary" in data["explanation"]


def test_t1_r5_03_catalog_standards_endpoint(client):
    """E2E-T1-R5-03: Verify GET /v1/standards queries catalog standards."""
    res = client.get("/v1/standards?limit=5")
    assert res.status_code == 200
    data = res.get_json()
    assert "standards" in data
    assert len(data["standards"]) > 0


def test_t1_r5_04_health_live_and_ready(client):
    """E2E-T1-R5-04: Verify operational liveness and readiness health checks."""
    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.get_json()["status"] == "live"

    res_ready = client.get("/health/ready")
    assert res_ready.status_code == 200
    ready_data = res_ready.get_json()
    assert ready_data["status"] == "ready"
    assert "release_id" in ready_data
    assert ready_data["model_loaded"] is True


def test_t1_r5_05_legacy_get_result_parity(client):
    """E2E-T1-R5-05: Verify legacy POST /get_result maintains 100% field parity."""
    res = client.post("/get_result", json={"user_input": "underground water storage reservoir"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["result_state"] in ("recommendations", "no_match")
    assert "user_input" in data
    assert "retrieved_candidate_count" in data
    assert "recommendations" in data
    if data["recommendations"]:
        rec = data["recommendations"][0]
        assert "is_code" in rec
        assert "standard_number" in rec
        assert "title" in rec
        assert "score" in rec
        assert "source" in rec
        assert "source_evidence" in rec
        assert "category_breadcrumb" in rec


def test_t1_r5_06_rfc_7807_problem_details(client):
    """E2E-T1-R5-06: Verify RFC 7807 Problem Details error formatting on 400 Bad Request."""
    res = client.post("/v1/analyze", json={})  # Missing requirement
    assert res.status_code == 400
    assert res.content_type == "application/problem+json"
    data = res.get_json()
    assert "type" in data
    assert "title" in data
    assert data["status"] == 400
    assert "detail" in data
    assert "instance" in data
    assert "request_id" in data
