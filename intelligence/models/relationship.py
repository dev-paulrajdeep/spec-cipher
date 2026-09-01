"""Read-only relationship traversal output backed by corpus relationship facts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelatedStandard:
    standard_number: str
    relationship_type: str
    source_standard_number: str
    page_number: int
    evidence_text: str
    depth: int
