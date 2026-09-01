"""Evidence is a direct view of a retrieved corpus chunk and never fabricated."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    standard_number: str
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    section_number: str | None
    section_title: str | None
    text: str
    relevance_score: float
