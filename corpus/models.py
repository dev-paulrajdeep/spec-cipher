"""Stable, serialisable models shared by all corpus stages.

The ingestion package intentionally keeps its raw extraction models.  These
models describe the richer, downstream canonical representation instead of
duplicating those ingestion-only types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BoundingBox:
    """A rectangle in PDF page coordinates (points, origin at top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BoundingBox must have non-negative width and height")


@dataclass(frozen=True)
class ElementProvenance:
    """Source-to-corpus links retained for text and non-text page elements."""

    source_file: str
    standard_id: str | None
    page_number: int
    section_numbers: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextBlock:
    """A raw text block with its physical location on a source page."""

    id: str
    standard_id: str | None
    page_number: int
    bbox: BoundingBox
    type: Literal["text"]
    content: str
    provenance: ElementProvenance


@dataclass(frozen=True)
class Table:
    """A deterministically detected table and a text projection for retrieval."""

    id: str
    standard_id: str | None
    page_number: int
    bbox: BoundingBox
    type: Literal["table"]
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    content: str
    provenance: ElementProvenance


@dataclass(frozen=True)
class Figure:
    """An embedded raster image or a vector figure represented by a page render."""

    id: str
    standard_id: str | None
    page_number: int
    bbox: BoundingBox
    type: Literal["image", "vector_figure"]
    asset_path: str
    provenance: ElementProvenance
    content: str | None = None


@dataclass(frozen=True)
class CorpusPage:
    """A source page with both the original extraction and cleaned text."""

    page_number: int
    raw_text: str
    cleaned_text: str
    ocr_applied: bool = False
    # The fields below are optional defaults to retain the original, small page
    # constructor contract for non-PDF sources and existing callers.
    id: str | None = None
    standard_id: str | None = None
    bbox: BoundingBox | None = None
    type: Literal["page"] = "page"
    width: float | None = None
    height: float | None = None
    asset_path: str | None = None
    provenance: ElementProvenance | None = None
    text_blocks: tuple[TextBlock, ...] = ()
    tables: tuple[Table, ...] = ()
    figures: tuple[Figure, ...] = ()


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
    # Optional ingestion provenance retained for corpus update / audit trails.
    source_path: str | None = None
    document_fingerprint: str | None = None


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
    # Kept out of the stable JSONL projection so existing retrieval indexes can
    # read both old and new artifacts. Canonical JSON retains these links.
    element_ids: tuple[str, ...] = ()
    source_path: str | None = None
    document_fingerprint: str | None = None
    document_id: str | None = None
    title: str | None = None
    chunking_version: str | None = None


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
    raw_asset_directory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible canonical data without implementation details."""
        data = asdict(self)
        for section in data["sections"]:
            section.pop("fragments", None)
        return _json_compatible(data)


def _json_compatible(value: Any) -> Any:
    """Convert dataclass containers to the JSON array/object primitives."""
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value
