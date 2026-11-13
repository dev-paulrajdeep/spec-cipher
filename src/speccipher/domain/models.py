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
    relative_path: str | None = None
    part_index: int | None = None
    page_offset: int = 0

    def __post_init__(self) -> None:
        if not self.source_file_id:
            raise ValueError("source_file_id must not be empty")
        if (
            not isinstance(self.page_count, int)
            or isinstance(self.page_count, bool)
            or self.page_count < 0
        ):
            raise ValueError("page_count must be non-negative")
        if self.part_index is not None and (
            not isinstance(self.part_index, int)
            or isinstance(self.part_index, bool)
            or self.part_index < 1
        ):
            raise ValueError("part_index must be a positive integer when provided")
        if (
            not isinstance(self.page_offset, int)
            or isinstance(self.page_offset, bool)
            or self.page_offset < 0
        ):
            raise ValueError("page_offset must be a non-negative integer")

    def absolute_page_for(self, physical_page: int) -> int:
        """Map a one-based source page to its continuous publication page."""
        if (
            not isinstance(physical_page, int)
            or isinstance(physical_page, bool)
            or physical_page < 1
            or physical_page > self.page_count
        ):
            raise ValueError("physical_page must be within the source file page range")
        return self.page_offset + physical_page

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file_id": self.source_file_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "checksum_sha256": self.checksum_sha256,
            "file_size_bytes": self.file_size_bytes,
            "relative_path": self.relative_path,
            "part_index": self.part_index,
            "page_offset": self.page_offset,
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
            relative_path=data.get("relative_path"),
            part_index=int(data["part_index"]) if data.get("part_index") is not None else None,
            page_offset=int(data.get("page_offset", 0)),
        )


@dataclass(frozen=True)
class MissingVolume:
    """Declared publication range with no ingested source file."""

    volume_id: str
    absolute_page_start: int
    absolute_page_end: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.volume_id, str) or not self.volume_id.strip():
            raise ValueError("volume_id must not be empty")
        if (
            not isinstance(self.absolute_page_start, int)
            or isinstance(self.absolute_page_start, bool)
            or self.absolute_page_start < 1
        ):
            raise ValueError("absolute_page_start must be positive")
        if (
            not isinstance(self.absolute_page_end, int)
            or isinstance(self.absolute_page_end, bool)
            or self.absolute_page_end < self.absolute_page_start
        ):
            raise ValueError("absolute_page_end cannot precede absolute_page_start")

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "absolute_page_start": self.absolute_page_start,
            "absolute_page_end": self.absolute_page_end,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissingVolume:
        return cls(
            volume_id=str(data["volume_id"]),
            absolute_page_start=int(data["absolute_page_start"]),
            absolute_page_end=int(data["absolute_page_end"]),
            reason=data.get("reason"),
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
    missing_volumes: tuple[MissingVolume, ...] = ()

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
            "missing_volumes": [volume.to_dict() for volume in self.missing_volumes],
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
            missing_volumes=tuple(
                MissingVolume.from_dict(volume) if isinstance(volume, dict) else volume
                for volume in data.get("missing_volumes", ())
            ),
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
    physical_page_start: int | None = None
    physical_page_end: int | None = None
    printed_page_label: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if (
            not isinstance(self.absolute_page_start, int)
            or isinstance(self.absolute_page_start, bool)
            or self.absolute_page_start < 1
        ):
            raise ValueError("absolute_page_start must be positive")
        if (
            not isinstance(self.absolute_page_end, int)
            or isinstance(self.absolute_page_end, bool)
            or self.absolute_page_end < self.absolute_page_start
        ):
            raise ValueError("absolute_page_end cannot be smaller than absolute_page_start")
        if (self.physical_page_start is None) != (self.physical_page_end is None):
            raise ValueError("physical page range must be fully known or fully unknown")
        if self.physical_page_start is not None and self.physical_page_end is not None:
            if (
                not isinstance(self.physical_page_start, int)
                or isinstance(self.physical_page_start, bool)
                or self.physical_page_start < 1
            ):
                raise ValueError("physical_page_start must be positive")
            if (
                not isinstance(self.physical_page_end, int)
                or isinstance(self.physical_page_end, bool)
                or self.physical_page_end < self.physical_page_start
            ):
                raise ValueError("physical_page_end cannot be smaller than physical_page_start")
        if self.printed_page_label is not None and not isinstance(self.printed_page_label, str):
            raise ValueError("printed_page_label must be a string when provided")

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
            "physical_page_start": self.physical_page_start,
            "physical_page_end": self.physical_page_end,
            "printed_page_label": self.printed_page_label,
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
            physical_page_start=(
                int(data["physical_page_start"])
                if data.get("physical_page_start") is not None
                else None
            ),
            physical_page_end=(
                int(data["physical_page_end"])
                if data.get("physical_page_end") is not None
                else None
            ),
            printed_page_label=data.get("printed_page_label"),
        )
