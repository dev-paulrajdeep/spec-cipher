from __future__ import annotations

import json
import sqlite3

import pytest

from speccipher.catalog.sqlite import CatalogRepository
from speccipher.offline.indexing.fts_build import (
    CatalogBuildError,
    build_catalog,
    build_catalog_payloads,
)


def _records() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    documents = [
        {
            "id": "doc_sp7",
            "family": "SP",
            "number": "7",
            "year": 2026,
            "code": "SP 7:2026",
            "title": "Building Code",
            "status": "current",
            "is_eligible": 1,
        }
    ]
    source_files = [
        {
            "id": "sp7_vol2",
            "logical_doc_id": "doc_sp7",
            "file_name": "volume-2.pdf",
            "relative_path": "sources/volume-2.pdf",
            "sha256": "a" * 64,
            "page_count": 400,
            "part_index": 2,
            "page_offset": 400,
        }
    ]
    chunks = [
        {
            "chunk_id": "doc_sp7_c1",
            "logical_doc_id": "doc_sp7",
            "standard_code": "SP 7:2026",
            "source_file_id": "sp7_vol2",
            "page_start": 10,
            "page_end": 11,
            "absolute_page_start": 410,
            "absolute_page_end": 411,
            "section_number": "5.3",
            "section_title": "Steel reinforcement",
            "text": "Steel reinforcement requirements for concrete buildings.",
            "is_eligible": 1,
        }
    ]
    return documents, source_files, chunks


def test_build_catalog_is_searchable_and_read_only_adapter_can_load(tmp_path):
    documents, source_files, chunks = _records()
    path = build_catalog(
        tmp_path / "catalog.sqlite",
        documents=documents,
        source_files=source_files,
        chunks=chunks,
        relationships=[
            {
                "source_code": "SP 7:2026",
                "target_code": "IS 1786:2008",
                "relationship_type": "references",
            }
        ],
    )

    catalog = CatalogRepository(path)
    try:
        found = catalog.search_fts("steel reinforcement", limit=5)
        assert [row["chunk_id"] for row in found] == ["doc_sp7_c1"]
        assert found[0]["absolute_page_start"] == 410
        assert catalog.get_standard_by_code("SP 7:2026")["title"] == "Building Code"
        assert catalog.get_relationships("SP 7:2026")[0]["target_code"] == "IS 1786:2008"
        with pytest.raises(sqlite3.OperationalError):
            catalog.conn.execute("DELETE FROM chunks")
    finally:
        catalog.close()


def test_catalog_rejects_absolute_pages_that_disagree_with_source_offset(tmp_path):
    documents, source_files, chunks = _records()
    chunks[0]["absolute_page_start"] = 409

    with pytest.raises(CatalogBuildError, match="source offset mapping"):
        build_catalog(
            tmp_path / "catalog.sqlite",
            documents=documents,
            source_files=source_files,
            chunks=chunks,
        )
    assert not (tmp_path / "catalog.sqlite").exists()


def test_catalog_refuses_to_overwrite_existing_artifact(tmp_path):
    target = tmp_path / "catalog.sqlite"
    target.write_bytes(b"keep me")
    documents, source_files, chunks = _records()

    with pytest.raises(FileExistsError):
        build_catalog(target, documents=documents, source_files=source_files, chunks=chunks)

    assert target.read_bytes() == b"keep me"


def test_catalog_rejects_chunk_linked_to_a_different_document(tmp_path):
    documents, source_files, chunks = _records()
    chunks[0]["logical_doc_id"] = "another_document"

    with pytest.raises(CatalogBuildError, match="different documents"):
        build_catalog(
            tmp_path / "catalog.sqlite",
            documents=documents,
            source_files=source_files,
            chunks=chunks,
        )
    assert not (tmp_path / "catalog.sqlite").exists()


def test_catalog_rejects_eligible_standard_stub(tmp_path):
    documents, source_files, chunks = _records()
    documents[0]["is_stub"] = 1

    with pytest.raises(CatalogBuildError, match="stub.*cannot be recommendation-eligible"):
        build_catalog(
            tmp_path / "catalog.sqlite",
            documents=documents,
            source_files=source_files,
            chunks=chunks,
        )
    assert not (tmp_path / "catalog.sqlite").exists()


def test_catalog_rejects_eligible_chunk_for_ineligible_document(tmp_path):
    documents, source_files, chunks = _records()
    documents[0]["is_eligible"] = 0

    with pytest.raises(CatalogBuildError, match="document is ineligible"):
        build_catalog(
            tmp_path / "catalog.sqlite",
            documents=documents,
            source_files=source_files,
            chunks=chunks,
        )
    assert not (tmp_path / "catalog.sqlite").exists()


def test_catalog_payload_builder_writes_chunk_jsonl_in_index_order(tmp_path):
    documents, source_files, chunks = _records()
    catalog_path, chunks_path = build_catalog_payloads(
        tmp_path / "catalog.sqlite",
        tmp_path / "chunks.jsonl",
        documents=documents,
        source_files=source_files,
        chunks=chunks,
    )

    assert catalog_path.is_file()
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert [row["chunk_id"] for row in rows] == ["doc_sp7_c1"]
    assert rows[0]["text"] == chunks[0]["text"]


def test_catalog_payload_builder_removes_partial_chunks_file_on_failure(tmp_path):
    documents, source_files, chunks = _records()
    chunks[0]["absolute_page_start"] = 409

    with pytest.raises(CatalogBuildError, match="source offset mapping"):
        build_catalog_payloads(
            tmp_path / "catalog.sqlite",
            tmp_path / "chunks.jsonl",
            documents=documents,
            source_files=source_files,
            chunks=chunks,
        )
    assert not (tmp_path / "catalog.sqlite").exists()
    assert not (tmp_path / "chunks.jsonl").exists()


def test_catalog_preserves_unresolved_identity_and_audit_records(tmp_path):
    documents = [{
        "id": "doc_unknown", "code": None, "title": None,
        "identity_status": "unresolved", "is_eligible": 0, "page_count": 2,
    }]
    source_files = [{
        "id": "source_unknown", "logical_doc_id": "doc_unknown",
        "page_count": 2, "page_offset": 0,
    }]
    chunks = [{
        "chunk_id": "doc_unknown_c00001", "logical_doc_id": "doc_unknown",
        "standard_code": None, "source_file_id": "source_unknown",
        "page_start": 1, "page_end": 2, "absolute_page_start": 1,
        "absolute_page_end": 2, "text": "Unresolved source evidence", "is_eligible": 0,
    }]
    path = build_catalog(
        tmp_path / "catalog.sqlite", documents=documents, source_files=source_files,
        chunks=chunks,
        operator_overrides=[{
            "override_id": "override-1", "logical_doc_id": "doc_unknown",
            "source_file_id": "source_unknown", "asserted_identity": "SP 7:2026",
            "operator_id": "operator-1", "timestamp": "2026-11-12T09:30:00Z",
            "reason": "Verified from publication index.",
            "evidence_sources": ["publication-index.pdf"],
        }],
        missing_volumes=[{
            "missing_volume_id": "missing-2", "logical_doc_id": "doc_unknown",
            "volume_id": "volume-2", "absolute_page_start": 401,
            "absolute_page_end": 800, "reason": "source unavailable",
        }],
    )
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT code, identity_status FROM logical_documents WHERE id = 'doc_unknown'"
        ).fetchone() == (None, "unresolved")
        assert connection.execute(
            "SELECT standard_code FROM chunks WHERE chunk_id = 'doc_unknown_c00001'"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT evidence_sources_json FROM operator_overrides WHERE override_id = 'override-1'"
        ).fetchone() == ('[\"publication-index.pdf\"]',)
        assert connection.execute(
            "SELECT absolute_page_end FROM missing_volumes WHERE missing_volume_id = 'missing-2'"
        ).fetchone() == (800,)
    finally:
        connection.close()
