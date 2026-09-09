"""Read and validate retrieval-ready corpus chunks; never read PDFs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from corpus.models import DocumentChunk
from corpus.processing.normalizer import normalize_standard_number


class CorpusChunkLoadError(ValueError):
    """Raised when a corpus artifact cannot safely be indexed."""


@dataclass(frozen=True)
class LoadedCorpus:
    """Validated chunks plus the source corpus schema version when available."""

    chunks: tuple[DocumentChunk, ...]
    schema_version: str | None


_REQUIRED_FIELDS = (
    "chunk_id", "standard_number", "source_file", "page_start", "page_end",
    "section_number", "section_title", "text",
)


class CorpusChunkLoader:
    """Load the exact JSON and JSONL contract emitted by ``corpus.serialization``."""

    def load(self, path: str | Path) -> LoadedCorpus:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Corpus artifact not found: {source}")
        if not source.is_file():
            raise CorpusChunkLoadError(f"Corpus artifact is not a file: {source}")
        if source.suffix.lower() == ".jsonl":
            rows = self._load_jsonl(source)
            schema_version = None
        elif source.suffix.lower() == ".json":
            rows, schema_version = self._load_json(source)
        else:
            raise CorpusChunkLoadError("Corpus artifact must be canonical .json or chunk .jsonl")
        chunks = tuple(self._chunk_from_row(row, f"record {index + 1}") for index, row in enumerate(rows))
        self._validate_unique_ids(chunks)
        return LoadedCorpus(chunks=chunks, schema_version=schema_version)

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusChunkLoadError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise CorpusChunkLoadError(f"JSONL line {line_number} must be an object")
            rows.append(row)
        return rows

    @staticmethod
    def _load_json(path: Path) -> tuple[list[dict[str, Any]], str | None]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusChunkLoadError(f"Invalid canonical JSON: {exc.msg}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
            raise CorpusChunkLoadError("Canonical JSON must be an object containing a chunks array")
        schema_version = payload.get("schema_version")
        if schema_version is not None and not isinstance(schema_version, str):
            raise CorpusChunkLoadError("schema_version must be a string when present")
        rows = payload["chunks"]
        if not all(isinstance(row, dict) for row in rows):
            raise CorpusChunkLoadError("Canonical JSON chunks must all be objects")
        metadata = payload.get("metadata")
        parent_standard_number = metadata.get("standard_number") if isinstance(metadata, dict) else None
        if not parent_standard_number and isinstance(metadata, dict):
            parent_standard_number = normalize_standard_number(str(metadata.get("source_file", "")))
        if parent_standard_number is not None and (
            not isinstance(parent_standard_number, str) or not parent_standard_number.strip()
        ):
            raise CorpusChunkLoadError("Canonical metadata.standard_number must be a non-empty string or null")
        if parent_standard_number:
            rows = [
                {
                    **row,
                    "standard_number": (
                        row.get("standard_number")
                        if row.get("standard_number") is not None
                        else parent_standard_number
                    ),
                }
                for row in rows
            ]
        return rows, schema_version

    @staticmethod
    def _chunk_from_row(row: dict[str, Any], location: str) -> DocumentChunk:
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise CorpusChunkLoadError(f"{location} is missing required fields: {', '.join(missing)}")
        string_fields = ("chunk_id", "source_file", "text")
        for field in string_fields:
            if not isinstance(row[field], str) or not row[field].strip():
                raise CorpusChunkLoadError(f"{location}.{field} must be a non-empty string")
        if row["standard_number"] is not None and (
            not isinstance(row["standard_number"], str) or not row["standard_number"].strip()
        ):
            raise CorpusChunkLoadError(f"{location}.standard_number must be a non-empty string or null")
        for field in ("page_start", "page_end"):
            if not isinstance(row[field], int) or isinstance(row[field], bool):
                raise CorpusChunkLoadError(f"{location}.{field} must be an integer")
        if row["page_start"] < 1 or row["page_end"] < row["page_start"]:
            raise CorpusChunkLoadError(f"{location} has an invalid page range")
        for field in ("section_number", "section_title"):
            if row[field] is not None and not isinstance(row[field], str):
                raise CorpusChunkLoadError(f"{location}.{field} must be a string or null")
        element_ids = row.get("element_ids", [])
        if (
            not isinstance(element_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in element_ids)
        ):
            raise CorpusChunkLoadError(f"{location}.element_ids must be a list of non-empty strings")
        standard_number = row["standard_number"] or normalize_standard_number(row["source_file"])
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            standard_number=standard_number,
            source_file=row["source_file"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_number=row["section_number"],
            section_title=row["section_title"],
            text=row["text"],
            element_ids=tuple(element_ids),
        )

    @staticmethod
    def _validate_unique_ids(chunks: tuple[DocumentChunk, ...]) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise CorpusChunkLoadError("Corpus artifact contains duplicate chunk_id values")
