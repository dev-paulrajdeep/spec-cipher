from __future__ import annotations

import hashlib

import pytest

from speccipher.offline.assembly.records import (
    CanonicalRecordError,
    records_from_canonical_documents,
)


def _document(
    *,
    source_path: str,
    standard_number: str | None,
    identity_status: str,
    document_id: str,
    page_count: int = 2,
    eligible: bool = True,
) -> dict:
    fingerprint = hashlib.sha256(source_path.encode()).hexdigest()
    return {
        "metadata": {
            "source_path": source_path,
            "source_file": source_path.rsplit("/", 1)[-1],
            "document_fingerprint": fingerprint,
            "document_id": document_id,
            "standard_number": standard_number,
            "title": "Concrete Structures Standard" if standard_number else None,
            "year": 2000 if standard_number else None,
            "edition": "Third revision" if standard_number else None,
            "status": "current" if standard_number else None,
            "page_count": page_count,
            "identity_status": identity_status,
            "identity_source": "header" if identity_status == "resolved" else "unresolved",
            "document_type": "standard",
            "recommendation_eligible": eligible,
        },
        "validation": [],
        "pages": [
            {"extraction_status": "text_extracted"} for _ in range(page_count)
        ],
        "sections": [
            {"title": "Scope", "text": "This standard covers concrete structures."}
        ],
        "relationships": [
            {
                "source_standard_number": standard_number,
                "target_standard_number": "IS 1786:2008",
                "relationship_type": "references",
                "page_number": 1,
                "evidence_text": "Reinforcement shall conform to IS 1786:2008.",
            }
        ] if standard_number else [],
        "chunks": [
            {
                "chunk_id": f"old_{source_path}",
                "page_start": 1,
                "page_end": 2,
                "section_number": "1",
                "section_title": "Scope",
                "text": f"Evidence from {source_path}",
                "recommendation_eligible": eligible,
            }
        ],
    }


def test_projects_single_document_provenance_scope_and_standard_stub():
    records = records_from_canonical_documents(
        [
            _document(
                source_path="IS456/volume-1.pdf",
                standard_number="IS 456:2000",
                identity_status="resolved",
                document_id="is456_2000",
            )
        ]
    )

    primary = records["documents"][0]
    chunk = records["chunks"][0]
    assert primary["id"] == "pub_is_456_2000"
    assert primary["is_eligible"] == 1
    assert records["source_files"][0]["sha256"] == hashlib.sha256(
        b"IS456/volume-1.pdf"
    ).hexdigest()
    assert chunk["absolute_page_start"] == 1
    assert chunk["absolute_page_end"] == 2
    assert chunk["chunk_id"] == "pub_is_456_2000_c00001"
    assert records["scope_texts"][0]["scope"].startswith("This standard covers")
    stub = next(row for row in records["documents"] if row["is_stub"])
    assert stub["code"] == "IS 1786:2008"
    assert stub["is_eligible"] == 0


def test_override_assembles_headerless_volume_with_continuous_absolute_pages():
    first = _document(
        source_path="SP7/volume-1.pdf",
        standard_number="SP 7:2026",
        identity_status="resolved",
        document_id="sp7_2026",
    )
    second = _document(
        source_path="SP7/volume-2.pdf",
        standard_number=None,
        identity_status="unresolved",
        document_id="doc_headerless_volume",
        eligible=False,
    )
    overrides = {
        "SP7/volume-2.pdf": {
            "asserted_identity": "SP 7:2026",
            "operator_id": "release-operator",
            "timestamp": "2026-11-12T09:30:00Z",
            "reason": "Volume is a declared continuation of the publication.",
            "evidence_sources": ["officialBIS/SP7/volume-2.pdf", "volume table of contents"],
        }
    }
    records = records_from_canonical_documents(
        [first, second],
        volume_layouts={
            "SP7/volume-1.pdf": {
                "page_start": 1, "page_end": 2, "part_index": 1,
                "logical_document_id": "pub_sp7_2026",
            },
            "SP7/volume-2.pdf": {
                "page_start": 3, "page_end": 4, "part_index": 2,
                "logical_document_id": "pub_sp7_2026",
            },
        },
        operator_overrides=overrides,
    )

    assert len([row for row in records["documents"] if not row["is_stub"]]) == 1
    assert records["documents"][0]["id"] == "pub_sp7_2026"
    assert records["documents"][0]["is_eligible"] == 1
    assert [row["page_offset"] for row in records["source_files"]] == [0, 2]
    second_volume_chunk = records["chunks"][1]
    assert second_volume_chunk["absolute_page_start"] == 3
    assert second_volume_chunk["absolute_page_end"] == 4
    assert second_volume_chunk["is_eligible"] == 1
    assert records["operator_overrides"][0]["operator_id"] == "release-operator"
    assert records["operator_overrides"][0]["evidence_sources"]


def test_missing_volume_is_recorded_without_fabricating_chunks():
    first = _document(
        source_path="IS456/volume-1.pdf",
        standard_number="IS 456:2000",
        identity_status="resolved",
        document_id="is456_2000",
    )
    second = _document(
        source_path="IS456/volume-3.pdf",
        standard_number="IS 456:2000",
        identity_status="resolved",
        document_id="is456_2000",
    )
    records = records_from_canonical_documents(
        [first, second],
        volume_layouts={
            "IS456/volume-1.pdf": {
                "page_start": 1, "page_end": 2,
                "logical_document_id": "pub_is456_2000",
            },
            "IS456/volume-3.pdf": {
                "page_start": 5, "page_end": 6, "part_index": 3,
                "logical_document_id": "pub_is456_2000",
            },
        },
        missing_volumes={
            "IS 456:2000": [
                {"volume_id": "volume-2", "page_start": 3, "page_end": 4, "reason": "source unavailable"}
            ]
        },
    )

    assert len(records["chunks"]) == 2
    assert [row["absolute_page_start"] for row in records["chunks"]] == [1, 5]
    assert records["missing_volumes"] == [
        {
            "missing_volume_id": "pub_is456_2000_missing_2",
            "logical_doc_id": "pub_is456_2000",
            "volume_id": "volume-2",
            "absolute_page_start": 3,
            "absolute_page_end": 4,
            "reason": "source unavailable",
        }
    ]


def test_unresolved_identity_remains_uneligible_and_multipart_ranges_are_required():
    unresolved = _document(
        source_path="unknown/source.pdf",
        standard_number=None,
        identity_status="unresolved",
        document_id="doc_source_hash",
        eligible=False,
    )
    projected = records_from_canonical_documents([unresolved])
    assert projected["documents"][0]["code"] is None
    assert projected["documents"][0]["is_eligible"] == 0
    assert projected["chunks"][0]["standard_code"] is None

    with pytest.raises(CanonicalRecordError, match="needs a declared page range"):
        records_from_canonical_documents(
            [
                _document(
                    source_path="IS456/volume-1.pdf",
                    standard_number="IS 456:2000",
                    identity_status="resolved",
                    document_id="is456_2000",
                ),
                _document(
                    source_path="IS456/volume-2.pdf",
                    standard_number="IS 456:2000",
                    identity_status="resolved",
                    document_id="is456_2000",
                ),
            ]
        )


def test_same_standard_number_does_not_prove_two_files_are_one_publication():
    with pytest.raises(CanonicalRecordError, match="matching standard numbers alone"):
        records_from_canonical_documents(
            [
                _document(
                    source_path="IS456/first.pdf",
                    standard_number="IS 456:2000",
                    identity_status="resolved",
                    document_id="first-publication",
                ),
                _document(
                    source_path="IS456/second.pdf",
                    standard_number="IS 456:2000",
                    identity_status="resolved",
                    document_id="second-publication",
                ),
            ],
            volume_layouts={
                "IS456/first.pdf": {
                    "page_start": 1, "page_end": 2, "logical_document_id": "pub_first",
                },
                "IS456/second.pdf": {
                    "page_start": 3, "page_end": 4, "logical_document_id": "pub_second",
                },
            },
        )
