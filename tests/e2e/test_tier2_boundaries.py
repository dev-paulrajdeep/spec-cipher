"""Tier 2: Boundary & Corner Cases E2E Tests (30 Tests across R1-R5).

Requirement areas:
- R1: Engineering Foundation & Architecture Boundaries (6 tests)
- R2: Two-Plane Architecture & Corpus Release System (6 tests)
- R3: Multipart Document Identity & Corpus Provenance (6 tests)
- R4: Hybrid Retrieval & Decision Engine (6 tests)
- R5: API, Operational Capabilities & Backward Compatibility (6 tests)
"""

from __future__ import annotations

import ast
import sqlite3
import sys
from pathlib import Path

import pytest
from speccipher.api.app import create_app
from speccipher.config.settings import Settings
from speccipher.domain import (
    Candidate,
    Confidence,
    ConfidenceLevel,
    DocumentChunk,
    LogicalDocument,
    Recommendation,
    RelationshipRole,
    Section,
    SourceFile,
    StandardId,
    StandardRelationship,
    StandardStub,
)
from speccipher.release.manifest import (
    ReleaseIntegrityError,
    ReleaseNotFoundError,
    verify_release,
)
from speccipher.services.container import ApplicationContainer
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
    """E2E-T2-R1-01: Verify Layer 4 sibling modules never cross-import each other via AST analysis."""
    src_root = Path("src/speccipher")
    siblings = {"retrieval", "decision", "ingestion", "understanding", "explanation"}

    def get_imported_modules(py_path: Path) -> list[tuple[int, str]]:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((node.lineno, node.module))
        return imports

    # Part 1: Verify actual codebase under src/speccipher has zero sibling cross-imports
    violations: list[str] = []
    for sibling in siblings:
        sib_dir = src_root / sibling
        if not sib_dir.exists():
            continue
        for py_file in sib_dir.rglob("*.py"):
            for lineno, mod_name in get_imported_modules(py_file):
                for other in siblings:
                    if other == sibling:
                        continue
                    if mod_name == f"speccipher.{other}" or mod_name.startswith(f"speccipher.{other}."):
                        violations.append(
                            f"{py_file}:{lineno} forbidden cross-import of sibling 'speccipher.{other}'"
                        )
    assert violations == [], f"Forbidden sibling cross-imports detected: {violations}"

    # Part 2: Boundary rejection verification against synthetic violating AST
    synthetic_bad_code = """
import speccipher.decision
from speccipher.understanding.extractor import extract_features
"""
    tree = ast.parse(synthetic_bad_code)
    detected_violations = []
    current_sibling = "retrieval"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for other in siblings - {current_sibling}:
                    if alias.name == f"speccipher.{other}" or alias.name.startswith(f"speccipher.{other}."):
                        detected_violations.append(other)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for other in siblings - {current_sibling}:
                if node.module == f"speccipher.{other}" or node.module.startswith(f"speccipher.{other}."):
                    detected_violations.append(other)

    assert "decision" in detected_violations
    assert "understanding" in detected_violations


def test_t2_r1_02_domain_purity_rejection_rule():
    """E2E-T2-R1-02: Verify domain kernel purity invariant forbids third-party libraries via AST inspection."""
    domain_dir = Path("src/speccipher/domain")
    assert domain_dir.exists(), "src/speccipher/domain directory must exist"

    prohibited_libs = {"numpy", "flask", "faiss", "torch", "sentence_transformers", "pydantic", "werkzeug"}

    def inspect_imports(tree: ast.AST) -> set[str]:
        top_level_pkgs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_pkgs.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    top_level_pkgs.add(node.module.split(".")[0])
        return top_level_pkgs

    # Part 1: Verify actual domain modules contain ZERO prohibited or non-stdlib imports
    for py_file in domain_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        imported_pkgs = inspect_imports(tree)
        intersection = imported_pkgs.intersection(prohibited_libs)
        assert not intersection, f"{py_file.name} violates domain purity by importing: {intersection}"
        for pkg in imported_pkgs:
            if pkg in ("__future__", "speccipher"):
                continue
            assert pkg in sys.stdlib_module_names, (
                f"{py_file.name} imports non-stdlib package '{pkg}'"
            )

    # Part 2: Boundary rejection verification
    synthetic_impure_code = """
import numpy as np
from pydantic import BaseModel
import torch
import faiss
"""
    impure_tree = ast.parse(synthetic_impure_code)
    detected_impure = inspect_imports(impure_tree)
    assert prohibited_libs.intersection(detected_impure) == {"numpy", "pydantic", "torch", "faiss"}


def test_t2_r1_03_invalid_settings_from_env(monkeypatch):
    """E2E-T2-R1-03: Verify Settings.from_env() raises ValueError on malformed env values."""
    monkeypatch.setenv("SPEC_CIPHER_PORT", "not_a_number")
    with pytest.raises(ValueError, match="Invalid integer for SPEC_CIPHER_PORT"):
        Settings.from_env()

    monkeypatch.setenv("SPEC_CIPHER_PORT", "5000")
    monkeypatch.setenv("SPEC_CIPHER_TOP_K", "-5")
    with pytest.raises(ValueError, match="(?i)top_k must be positive"):
        Settings.from_env()


def test_t2_r1_04_non_existent_release_directory(tmp_path: Path):
    """E2E-T2-R1-04: Verify ApplicationContainer raises ReleaseNotFoundError on missing path."""
    missing_dir = tmp_path / "non_existent_release_dir"
    settings = Settings.from_env(release_path=missing_dir)
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
    settings = Settings.from_env(release_path=tmp_path)
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
    """E2E-T2-R3-01: Verify multipart assembly raises MultipartAssemblyError on non-contiguous page ranges."""
    from speccipher.domain.errors import MultipartAssemblyError
    from speccipher.ingestion.assembly.multipart import assemble_multipart_publication

    # Volume 1 ends at page 400, Volume 2 begins at page 501 (gap of 100 pages: 401-500 missing)
    vols = [
        {"filename": "SP 7_2026-001-400.pdf", "page_start": 1, "page_end": 400, "page_count": 400},
        {"filename": "SP 7_2026-501-800.pdf", "page_start": 501, "page_end": 800, "page_count": 300},
    ]

    with pytest.raises(MultipartAssemblyError, match="(?i)non-contiguous|gap"):
        assemble_multipart_publication(vols)


def test_t2_r3_02_headless_volume_without_override(container: ApplicationContainer):
    """E2E-T2-R3-02: Verify headless volume without override retains unresolved status."""
    cur = container.catalog.conn.cursor()
    cur.execute("SELECT is_eligible FROM logical_documents WHERE is_stub=1")
    row = cur.fetchone()
    assert row[0] == 0  # Not eligible


def test_t2_r3_03_malformed_operator_override_validation():
    """E2E-T2-R3-03: Verify operator override missing required audit fields raises OverrideValidationError."""
    from speccipher.domain.errors import OverrideValidationError
    from speccipher.ingestion.overrides.loader import validate_operator_override

    malformed_override = {
        "source_filename": "SP 7_2026-401-800.pdf",
    }

    with pytest.raises(OverrideValidationError, match="(?i)missing required.*operator"):
        validate_operator_override(malformed_override)

    missing_rationale = {
        "source_filename": "SP 7_2026-401-800.pdf",
        "operator": "lead-architect",
        "timestamp": "2026-11-11T08:00:00Z",
    }
    with pytest.raises(OverrideValidationError, match="(?i)rationale"):
        validate_operator_override(missing_rationale)


def test_t2_r3_04_out_of_range_page_number_detection():
    """E2E-T2-R3-04: Verify chunk reporting page exceeding source volume page count raises ProvenanceValidationError."""
    from speccipher.domain.errors import ProvenanceValidationError
    from speccipher.domain.identity import StandardId
    from speccipher.domain.models import DocumentChunk, SourceFile
    from speccipher.ingestion.validation import validate_chunk_provenance

    source = SourceFile(
        source_file_id="vol1.pdf",
        filename="vol1.pdf",
        page_count=400,
    )

    invalid_chunk = DocumentChunk(
        chunk_id="chk-out-of-bounds",
        standard_id=StandardId(family="SP", number="7", year=2026),
        logical_doc_id="SP_7_2026",
        source_file_id="vol1.pdf",
        absolute_page_start=450,  # Exceeds total source page count of 400
        absolute_page_end=452,
        section_number="2.1",
        section_title="Fire Safety",
        text="Sample text",
        is_eligible=True,
    )

    with pytest.raises(ProvenanceValidationError, match="(?i)page.*exceeds.*source"):
        validate_chunk_provenance(invalid_chunk, source)


def test_t2_r3_05_conflicting_identities_across_volumes():
    """E2E-T2-R3-05: Verify conflicting identities across multi-part volumes raises MultipartAssemblyError."""
    from speccipher.domain.errors import MultipartAssemblyError
    from speccipher.ingestion.assembly.multipart import assemble_multipart_publication

    conflicting_vols = [
        {
            "filename": "IS 456-vol1.pdf",
            "page_start": 1,
            "page_end": 100,
            "page_count": 100,
            "asserted_identity": "IS 456:2000",
        },
        {
            "filename": "IS 800-vol2.pdf",
            "page_start": 101,
            "page_end": 200,
            "page_count": 100,
            "asserted_identity": "IS 800:2007",
        },
    ]

    with pytest.raises(MultipartAssemblyError, match="(?i)conflicting.*identit"):
        assemble_multipart_publication(conflicting_vols)


def test_t2_r3_06_standard_stub_with_missing_title():
    """E2E-T2-R3-06: Verify StandardStub preserves title as None without inventing."""
    stub = StandardStub(
        standard_id=StandardId(family="IS", number="1234"),
        title=None,
    )
    assert stub.title is None
    assert stub.is_stub is True


# ============================================================================
# R4: Query & Ranking Boundaries, Extreme Lengths & Abstention (6 Tests)
# ============================================================================


def test_t2_r4_01_whitespace_only_query_abstention(client):
    """E2E-T2-R4-01: Verify whitespace-only query fails validation or returns 400."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_WHITESPACE})
    assert res.status_code == 400
    data = res.get_json()
    assert "requirement" in str(data)


def test_t2_r4_02_extreme_length_query_rejection(client):
    """E2E-T2-R4-02: Verify query exceeding 4000 characters is rejected with 400 Payload Too Large."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_EXTREME_LENGTH})
    assert res.status_code in (400, 413)
    data = res.get_json()
    assert "exceeds" in data.get("detail", "").lower() or "too large" in data.get("title", "").lower()


def test_t2_r4_03_zero_similarity_abstention(client):
    """E2E-T2-R4-03: Verify query with ungrounded vocabulary triggers explicit abstention."""
    res = client.post("/v1/analyze", json={"requirement": QUERY_ABSTENTION_RANDOM})
    assert res.status_code == 200
    data = res.get_json()
    assert data["result_state"] in ("abstain", "no_match")
    assert len(data.get("recommendations", [])) == 0


def test_t2_r4_04_single_token_query_handling(client):
    """E2E-T2-R4-04: Verify single-word query handles gracefully without crashing."""
    res = client.post("/v1/analyze", json={"requirement": "concrete"})
    assert res.status_code == 200
    data = res.get_json()
    assert "recommendations" in data


def test_t2_r4_05_withdrawn_standard_not_recommended(client):
    """E2E-T2-R4-05: Verify withdrawn/ineligible standard is excluded from recommendations."""
    res = client.post("/v1/analyze", json={"requirement": "specification for legacy testing standard"})
    assert res.status_code == 200
    data = res.get_json()
    recs = data.get("recommendations", [])
    for r in recs:
        assert r["standard_id"]["code"] != "IS 9999:2010"


def test_t2_r4_06_exact_score_tie_breaking():
    """E2E-T2-R4-06: Verify tie-breaking determinism in fusion engine when scores are identical."""
    from speccipher.retrieval.fusion import reciprocal_rank_fusion

    dense1 = [{"chunk_id": "chunk-beta"}, {"chunk_id": "chunk-alpha"}]
    lexical1 = [{"chunk_id": "chunk-alpha"}, {"chunk_id": "chunk-beta"}]

    res1 = reciprocal_rank_fusion(dense1, lexical1)
    assert res1[0]["rrf_score"] == res1[1]["rrf_score"]
    assert res1[0]["chunk_id"] == "chunk-alpha"
    assert res1[1]["chunk_id"] == "chunk-beta"

    dense2 = [{"chunk_id": "chunk-alpha"}, {"chunk_id": "chunk-beta"}]
    lexical2 = [{"chunk_id": "chunk-beta"}, {"chunk_id": "chunk-alpha"}]

    res2 = reciprocal_rank_fusion(dense2, lexical2)
    assert res2[0]["chunk_id"] == "chunk-alpha"
    assert res2[1]["chunk_id"] == "chunk-beta"


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
    settings = Settings.from_env(release_path=tmp_path)
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
