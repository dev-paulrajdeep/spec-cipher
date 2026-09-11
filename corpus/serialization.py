"""Canonical JSON and JSONL writers for corpus output."""

from __future__ import annotations

import json
from pathlib import Path

from corpus.models import DocumentChunk, StandardDocument


def write_document_json(document: StandardDocument, path: str | Path) -> None:
    """Write deterministic, human-inspectable canonical document JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_chunks_jsonl(chunks: tuple[DocumentChunk, ...] | list[DocumentChunk], path: str | Path) -> None:
    """Write one independently indexable retrieval chunk per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            row = {
                "chunk_id": chunk.chunk_id,
                "document_fingerprint": chunk.document_fingerprint,
                "document_id": chunk.document_id,
                "document_type": chunk.document_type,
                "identity_status": chunk.identity_status,
                "page_end": chunk.page_end,
                "page_start": chunk.page_start,
                "recommendation_eligible": chunk.recommendation_eligible,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "source_file": chunk.source_file,
                "source_path": chunk.source_path,
                "standard_number": chunk.standard_number,
                "text": chunk.text,
                "title": chunk.title,
                "chunking_version": chunk.chunking_version,
            }
            # Omit null optional provenance fields to keep older loaders tolerant
            # while retaining explicit null for standard_number/title/section fields.
            compact = {
                key: value
                for key, value in row.items()
                if value is not None
                or key
                in {
                    "standard_number",
                    "section_number",
                    "section_title",
                    "title",
                    "recommendation_eligible",
                    "document_type",
                    "identity_status",
                }
            }
            handle.write(json.dumps(compact, ensure_ascii=False, sort_keys=True) + "\n")
