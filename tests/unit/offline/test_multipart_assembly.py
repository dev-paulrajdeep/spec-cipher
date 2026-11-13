from __future__ import annotations

import pytest

from speccipher.domain.errors import MultipartAssemblyError
from speccipher.ingestion.assembly import assemble_multipart_publication


def test_orders_by_explicit_absolute_page_ranges_and_maps_physical_pages() -> None:
    result = assemble_multipart_publication(
        [
            {
                "filename": "volume-10.pdf",
                "page_start": 401,
                "page_end": 800,
                "page_count": 400,
            },
            {
                "filename": "volume-2.pdf",
                "page_start": 1,
                "page_end": 400,
                "page_count": 400,
            },
        ]
    )

    first, second = result["volumes"]
    assert result["status"] == "assembled"
    assert result["volumes_count"] == 2
    assert result["total_pages"] == result["declared_page_span"] == 800
    assert first["filename"] == "volume-2.pdf"
    assert first["volume_order"] == 1
    assert first["page_offset"] == 0
    assert first["physical_page_start"] == 1
    assert first["absolute_page_start"] == 1
    assert second["filename"] == "volume-10.pdf"
    assert second["volume_order"] == 2
    assert second["page_offset"] == 400
    assert second["physical_page_end"] == 400
    assert second["absolute_page_end"] == 800


def test_unexplained_gap_fails_closed() -> None:
    with pytest.raises(MultipartAssemblyError, match="Non-contiguous.*declare the missing volume"):
        assemble_multipart_publication(
            [
                {"filename": "first.pdf", "page_start": 1, "page_end": 400},
                {"filename": "third.pdf", "page_start": 801, "page_end": 1200},
            ]
        )


def test_explicit_missing_volume_is_reported_without_fabricating_a_source_volume() -> None:
    result = assemble_multipart_publication(
        [
            {"filename": "first.pdf", "page_start": 1, "page_end": 400},
            {"filename": "third.pdf", "page_start": 801, "page_end": 1200},
        ],
        missing_volumes=[
            {
                "volume_id": "volume-2",
                "page_start": 401,
                "page_end": 800,
                "reason": "source file not supplied",
            }
        ],
    )

    assert result["status"] == "partial"
    assert result["volumes_count"] == 2
    assert result["total_pages"] == 800
    assert result["declared_page_span"] == 1200
    assert [volume["filename"] for volume in result["volumes"]] == ["first.pdf", "third.pdf"]
    assert result["missing_volumes"] == [
        {
            "volume_id": "volume-2",
            "page_start": 401,
            "page_end": 800,
            "page_count": 400,
            "reason": "source file not supplied",
            "missing": True,
            "volume_order": 2,
        }
    ]


def test_page_count_must_match_declared_absolute_range() -> None:
    with pytest.raises(MultipartAssemblyError, match="page_count.*does not match"):
        assemble_multipart_publication([{"page_start": 1, "page_end": 400, "page_count": 399}])


def test_overlapping_ranges_fail_closed() -> None:
    with pytest.raises(MultipartAssemblyError, match="Overlapping declared page ranges"):
        assemble_multipart_publication(
            [
                {"page_start": 1, "page_end": 400},
                {"page_start": 400, "page_end": 800},
            ]
        )


def test_volume_without_declared_range_is_rejected_instead_of_filename_sorting() -> None:
    with pytest.raises(MultipartAssemblyError, match="page_start must be a positive integer"):
        assemble_multipart_publication([{"filename": "volume-1.pdf"}])


def test_conflicting_asserted_identities_fail_closed() -> None:
    with pytest.raises(MultipartAssemblyError, match="Conflicting publication identities"):
        assemble_multipart_publication(
            [
                {"page_start": 1, "page_end": 100, "asserted_identity": "IS 456:2000"},
                {"page_start": 101, "page_end": 200, "asserted_identity": "IS 800:2007"},
            ]
        )


def test_missing_declaration_requires_auditable_volume_id_and_range() -> None:
    with pytest.raises(MultipartAssemblyError, match="requires a non-empty volume_id"):
        assemble_multipart_publication(
            [
                {"page_start": 1, "page_end": 10},
                {"page_start": 21, "page_end": 30},
            ],
            missing_volumes=[{"page_start": 11, "page_end": 20}],
        )
