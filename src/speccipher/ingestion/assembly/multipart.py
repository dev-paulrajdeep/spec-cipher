"""Multipart publication assembly for SpecCipher (Layer 4 Ingestion)."""

from __future__ import annotations

from typing import Any

from speccipher.domain.errors import MultipartAssemblyError


def assemble_multipart_publication(volumes: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble multiple volume files into a unified logical publication.

    Enforces invariants:
    - Contiguous page ranges across sequentially ordered volumes.
    - Consistent asserted standard identities across volumes.
    """
    if not volumes:
        raise MultipartAssemblyError("Empty volume list cannot be assembled")

    sorted_vols = sorted(volumes, key=lambda v: v.get("page_start", 0))

    # Invariant 1: Contiguous page ranges
    for i in range(len(sorted_vols) - 1):
        expected_next = sorted_vols[i]["page_end"] + 1
        actual_next = sorted_vols[i + 1]["page_start"]
        if actual_next != expected_next:
            raise MultipartAssemblyError(
                f"Non-contiguous page range between volume {i+1} (ends at {sorted_vols[i]['page_end']}) "
                f"and volume {i+2} (starts at {actual_next}): gap of {actual_next - expected_next} pages"
            )

    # Invariant 2: Non-conflicting identities
    identities = {v["asserted_identity"] for v in sorted_vols if "asserted_identity" in v}
    if len(identities) > 1:
        raise MultipartAssemblyError(
            f"Conflicting publication identities across volumes: {sorted(identities)}"
        )

    total_pages = sum(
        v.get("page_count", v["page_end"] - v["page_start"] + 1) for v in sorted_vols
    )
    return {
        "status": "assembled",
        "volumes_count": len(sorted_vols),
        "total_pages": total_pages,
    }
