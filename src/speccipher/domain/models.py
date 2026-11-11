"""Pure domain models for documents, source files, sections, and retrieval chunks.

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .identity import StandardId


@dataclass(frozen=True)
class SourceFile:
    """Physical ingested PDF or document source file."""

    source_file_id: str
    filename: str
    page_count: int
    checksum_sha256: str | None = None
    file_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.source_file_id:
            raise ValueError("source_file_id must not be empty")
        if self.page_count < 0:
            raise ValueError("page_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file_id": self.source_file_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "checksum_sha256": self.checksum_sha256,
            "file_size_bytes": self.file_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceFile:
        return cls(
            source_file_id=str(data["source_file_id"]),
            filename=str(data["filename"]),
            page_count=int(data["page_count"]),
            checksum_sha256=data.get("checksum_sha256"),
            file_size_bytes=int(data["file_size_bytes"])
            if data.get("file_size_bytes") is not None
            else None,
        )


@dataclass(frozen=True)
class LogicalDocument:
    """Logical standard publication which may span one or more physical files."""

    logical_doc_id: str
    standard_id: StandardId
    title: str
    edition: str | None = None
    revision: str | None = None
    status: str | None = None
    is_multipart: bool = False
    source_file_ids: tuple[str, ...] = ()
    total_pages: int = 0
    recommendation_eligible: bool = True

    def __post_init__(self) -> None:
        if not self.logical_doc_id:
            raise ValueError("logical_doc_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_doc_id": self.logical_doc_id,
            "standard_id": self.standard_id.to_dict(),
            "title": self.title,
            "edition": self.edition,
            "revision": self.revision,
            "status": self.status,
            "is_multipart": self.is_multipart,
            "source_file_ids": list(self.source_file_ids),
            "total_pages": self.total_pages,
            "recommendation_eligible": self.recommendation_eligible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogicalDocument:
        sid_raw = data["standard_id"]
        sid = StandardId.from_dict(sid_raw) if isinstance(sid_raw, (dict, str)) else sid_raw
        return cls(
            logical_doc_id=str(data["logical_doc_id"]),
            standard_id=sid,
            title=str(data["title"]),
            edition=data.get("edition"),
            revision=data.get("revision"),
            status=data.get("status"),
            is_multipart=bool(data.get("is_multipart", False)),
            source_file_ids=tuple(data.get("source_file_ids", ())),
            total_pages=int(data.get("total_pages", 0)),
            recommendation_eligible=bool(data.get("recommendation_eligible", True)),
        )


@dataclass(frozen=True)
class Section:
    """Structural section detected within a standard."""

    section_number: str | None = None
    title: str | None = None
    level: int = 1
    page_start: int = 1
    page_end: int = 1
    text: str = ""
    section_id: str | None = None

    def __post_init__(self) -> None:
        if self.page_end < self.page_start:
            raise ValueError("page_end cannot be smaller than page_start")

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_number": self.section_number,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        return cls(
            section_id=data.get("section_id"),
            section_number=data.get("section_number"),
            title=data.get("title"),
            level=int(data.get("level", 1)),
            page_start=int(data.get("page_start", 1)),
            page_end=int(data.get("page_end", 1)),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True)
class DocumentChunk:
    """Deterministic retrieval unit preserving source provenance and absolute page offsets.

    Includes backward-compatible property accessors (`standard_number`, `source_file`,
    `page_start`, `page_end`, `recommendation_eligible`) to seamlessly support legacy callers.
    """

    chunk_id: str
    standard_id: StandardId | None
    logical_doc_id: str
    source_file_id: str
    absolute_page_start: int
    absolute_page_end: int
    section_number: str | None = None
    section_title: str | None = None
    text: str = ""
    is_eligible: bool = True
    element_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if self.absolute_page_end < self.absolute_page_start:
            raise ValueError("absolute_page_end cannot be smaller than absolute_page_start")

    @property
    def standard_number(self) -> str | None:
        """Backward-compatible standard number string."""
        return self.standard_id.code if self.standard_id else None

    @property
    def source_file(self) -> str:
        """Backward-compatible source filename string."""
        return self.source_file_id

    @property
    def page_start(self) -> int:
        """Backward-compatible page start alias."""
        return self.absolute_page_start

    @property
    def page_end(self) -> int:
        """Backward-compatible page end alias."""
        return self.absolute_page_end

    @property
    def recommendation_eligible(self) -> bool:
        """Backward-compatible recommendation eligibility alias."""
        return self.is_eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "standard_id": self.standard_id.to_dict() if self.standard_id else None,
            "logical_doc_id": self.logical_doc_id,
            "source_file_id": self.source_file_id,
            "absolute_page_start": self.absolute_page_start,
            "absolute_page_end": self.absolute_page_end,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "text": self.text,
            "is_eligible": self.is_eligible,
            "element_ids": list(self.element_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentChunk:
        sid_raw = data.get("standard_id")
        sid: StandardId | None = None
        if isinstance(sid_raw, (dict, str)):
            sid = StandardId.from_dict(sid_raw)
        elif data.get("standard_number"):
            try:
                sid = StandardId.parse(str(data["standard_number"]))
            except Exception:
                sid = None

        return cls(
            chunk_id=str(data["chunk_id"]),
            standard_id=sid,
            logical_doc_id=str(data.get("logical_doc_id") or data.get("source_file_id") or "doc"),
            source_file_id=str(
                data.get("source_file_id") or data.get("source_file") or "source.pdf"
            ),
            absolute_page_start=int(data.get("absolute_page_start", data.get("page_start", 1))),
            absolute_page_end=int(data.get("absolute_page_end", data.get("page_end", 1))),
            section_number=data.get("section_number"),
            section_title=data.get("section_title"),
            text=str(data.get("text", "")),
            is_eligible=bool(data.get("is_eligible", data.get("recommendation_eligible", True))),
            element_ids=tuple(data.get("element_ids", ())),
            metadata=dict(data.get("metadata", {})),
        )
