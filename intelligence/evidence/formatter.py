"""Small presentation-neutral formatter for provenance citations."""

from __future__ import annotations

from intelligence.models import EvidenceItem


def format_evidence(item: EvidenceItem) -> str:
    section = item.section_number or item.section_title or "unsectioned"
    pages = str(item.page_start) if item.page_start == item.page_end else f"{item.page_start}-{item.page_end}"
    return f"{item.standard_number}, {section}, page {pages}, chunk {item.chunk_id}"
