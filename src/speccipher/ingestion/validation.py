"""Provenance and chunk boundary validation (Layer 4 Ingestion)."""

from __future__ import annotations

from typing import Any

from speccipher.domain.errors import ProvenanceValidationError


def validate_chunk_provenance(chunk: Any, source_file: Any) -> None:
    """Validate that chunk page boundaries are within source file boundaries."""
    page_start = getattr(chunk, "absolute_page_start", None)
    page_end = getattr(chunk, "absolute_page_end", None)

    if page_start is None or page_end is None:
        raise ProvenanceValidationError("Chunk missing absolute page start or end")

    if page_start < 1:
        raise ProvenanceValidationError(f"Invalid chunk page start: {page_start} < 1")

    if page_end < page_start:
        raise ProvenanceValidationError(
            f"Invalid chunk page range: end ({page_end}) < start ({page_start})"
        )

    file_page_count = getattr(source_file, "page_count", 0)
    filename = getattr(source_file, "filename", getattr(source_file, "file_name", "unknown"))

    if page_start > file_page_count or page_end > file_page_count:
        raise ProvenanceValidationError(
            f"Chunk page range [{page_start}, {page_end}] exceeds source file '{filename}' "
            f"page count {file_page_count}"
        )
