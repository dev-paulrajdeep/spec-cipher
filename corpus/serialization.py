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


def write_chunks_jsonl(chunks: tuple[DocumentChunk, ...], path: str | Path) -> None:
    """Write one independently indexable retrieval chunk per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            row = {
                "chunk_id": chunk.chunk_id,
                "standard_number": chunk.standard_number,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "text": chunk.text,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
