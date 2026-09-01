"""Stable, serialisable models shared by all corpus stages.

The ingestion package intentionally keeps its raw extraction models.  These
models describe the richer, downstream canonical representation instead of
duplicating those ingestion-only types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorpusPage:
    """A source page with both the original extraction and cleaned text."""

    page_number: int
    raw_text: str
    cleaned_text: str
    ocr_applied: bool = False


@dataclass(frozen=True)
class StandardMetadata:
    """Only values evidenced by the source; unknown fields remain ``None``."""

    standard_number: str | None
    title: str | None
    year: int | None
    edition: str | None
    revision: str | None
    status: str | None
    source_file: str
    page_count: int


@dataclass(frozen=True)
class SectionFragment:
    """The portion of a section originating on one physical source page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Section:
    """A conservatively detected structural section of a standard."""

    section_number: str | None
    title: str | None
    level: int
    page_start: int
    page_end: int
    text: str
    fragments: tuple[SectionFragment, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class StandardRelationship:
    """An explicit source-backed relationship for a later graph/traversal layer."""

    source_standard_number: str | None
    target_standard_number: str
    relationship_type: str
    page_number: int
    evidence_text: str


@dataclass(frozen=True)
class DocumentChunk:
    """A deterministic retrieval unit, retaining its complete provenance."""

    chunk_id: str
    standard_number: str | None
    source_file: str
    page_start: int
    page_end: int
    section_number: str | None
    section_title: str | None
    text: str


@dataclass(frozen=True)
class ValidationIssue:
    """A validation finding; severity is ``error``, ``warning``, or ``uncertain``."""

    severity: str
    code: str
    message: str
    page_number: int | None = None


@dataclass(frozen=True)
class StandardDocument:
    """Canonical corpus record handed to the future retrieval layer."""

    schema_version: str
    metadata: StandardMetadata
    pages: tuple[CorpusPage, ...]
    sections: tuple[Section, ...]
    relationships: tuple[StandardRelationship, ...]
    chunks: tuple[DocumentChunk, ...]
    validation: tuple[ValidationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible canonical data without implementation details."""
        data = asdict(self)
        for section in data["sections"]:
            section.pop("fragments", None)
        return data
