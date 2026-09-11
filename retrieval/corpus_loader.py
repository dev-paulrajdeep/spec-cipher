"""Read and validate retrieval-ready corpus chunks; never read PDFs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from corpus.models import DocumentChunk


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
        # Never invent identity from filenames at load time. Filename may only
        # fill gaps when the canonical metadata already carries an explicit number.
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
        # Preserve explicit null. Do not synthesize identity from source_file.
        standard_number = row["standard_number"]
        source_path = row.get("source_path")
        document_fingerprint = row.get("document_fingerprint")
        if source_path is not None and (not isinstance(source_path, str) or not source_path.strip()):
            raise CorpusChunkLoadError(f"{location}.source_path must be a non-empty string or null")
        if document_fingerprint is not None and (
            not isinstance(document_fingerprint, str) or not document_fingerprint.strip()
        ):
            raise CorpusChunkLoadError(
                f"{location}.document_fingerprint must be a non-empty string or null"
            )
        document_id = row.get("document_id")
        title = row.get("title")
        chunking_version = row.get("chunking_version")
        document_type = row.get("document_type")
        identity_status = row.get("identity_status")
        recommendation_eligible = row.get("recommendation_eligible")
        for optional_name, optional_value in (
            ("document_id", document_id),
            ("title", title),
            ("chunking_version", chunking_version),
            ("document_type", document_type),
            ("identity_status", identity_status),
        ):
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise CorpusChunkLoadError(
                    f"{location}.{optional_name} must be a non-empty string or null"
                )
        if recommendation_eligible is not None and not isinstance(recommendation_eligible, bool):
            raise CorpusChunkLoadError(
                f"{location}.recommendation_eligible must be a boolean or null"
            )
        if identity_status is not None and identity_status not in {"resolved", "unresolved"}:
            raise CorpusChunkLoadError(
                f"{location}.identity_status must be 'resolved', 'unresolved', or null"
            )
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
            source_path=source_path,
            document_fingerprint=document_fingerprint,
            document_id=document_id,
            title=title,
            chunking_version=chunking_version,
            document_type=document_type,
            recommendation_eligible=recommendation_eligible,
            identity_status=identity_status,
        )

    @staticmethod
    def _validate_unique_ids(chunks: tuple[DocumentChunk, ...]) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise CorpusChunkLoadError("Corpus artifact contains duplicate chunk_id values")
