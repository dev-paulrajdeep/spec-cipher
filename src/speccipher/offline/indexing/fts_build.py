"""Build the SQLite catalog and FTS5 index used by serving releases."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CatalogRow = Mapping[str, Any]

_SCHEMA = """
CREATE TABLE logical_documents (
    id TEXT PRIMARY KEY,
    family TEXT,
    number TEXT,
    part TEXT,
    year INTEGER,
    code TEXT UNIQUE,
    title TEXT,
    status TEXT,
    edition TEXT,
    identity_source TEXT,
    identity_status TEXT NOT NULL DEFAULT 'verified',
    is_eligible INTEGER NOT NULL DEFAULT 1 CHECK (is_eligible IN (0, 1)),
    is_stub INTEGER NOT NULL DEFAULT 0 CHECK (is_stub IN (0, 1)),
    page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0)
);
CREATE TABLE source_files (
    id TEXT PRIMARY KEY,
    logical_doc_id TEXT NOT NULL REFERENCES logical_documents(id),
    file_name TEXT,
    relative_path TEXT,
    sha256 TEXT,
    page_count INTEGER NOT NULL CHECK (page_count >= 0),
    part_index INTEGER,
    page_offset INTEGER NOT NULL DEFAULT 0 CHECK (page_offset >= 0),
    extraction_status TEXT NOT NULL DEFAULT 'processed'
);
CREATE TABLE sections (
    id TEXT PRIMARY KEY,
    logical_doc_id TEXT NOT NULL REFERENCES logical_documents(id),
    section_number TEXT,
    section_title TEXT,
    level INTEGER NOT NULL DEFAULT 1,
    page_start INTEGER,
    page_end INTEGER,
    text TEXT
);
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_code TEXT NOT NULL,
    target_code TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    page_number INTEGER,
    evidence_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE scope_text (
    standard_code TEXT PRIMARY KEY,
    scope TEXT,
    exclusions TEXT
);
CREATE TABLE operator_overrides (
    override_id TEXT PRIMARY KEY,
    logical_doc_id TEXT NOT NULL REFERENCES logical_documents(id),
    source_file_id TEXT NOT NULL REFERENCES source_files(id),
    asserted_identity TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_sources_json TEXT NOT NULL
);
CREATE TABLE missing_volumes (
    missing_volume_id TEXT PRIMARY KEY,
    logical_doc_id TEXT NOT NULL REFERENCES logical_documents(id),
    volume_id TEXT NOT NULL,
    absolute_page_start INTEGER NOT NULL CHECK (absolute_page_start >= 1),
    absolute_page_end INTEGER NOT NULL CHECK (absolute_page_end >= absolute_page_start),
    reason TEXT
);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    logical_doc_id TEXT NOT NULL REFERENCES logical_documents(id),
    standard_code TEXT,
    source_file_id TEXT NOT NULL REFERENCES source_files(id),
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    absolute_page_start INTEGER NOT NULL,
    absolute_page_end INTEGER NOT NULL,
    section_number TEXT,
    section_title TEXT,
    text TEXT NOT NULL,
    is_eligible INTEGER NOT NULL DEFAULT 1 CHECK (is_eligible IN (0, 1))
);
CREATE INDEX chunks_by_document ON chunks(logical_doc_id, chunk_id);
CREATE INDEX chunks_by_standard ON chunks(standard_code, chunk_id);
CREATE VIRTUAL TABLE chunks_fts USING fts5 (
    chunk_id UNINDEXED,
    standard_code,
    section_number,
    section_title,
    text,
    tokenize='unicode61'
);
"""


class CatalogBuildError(ValueError):
    """Raised when source records cannot form a provenance-complete catalog."""


def _required(row: CatalogRow, *keys: str) -> tuple[Any, ...]:
    missing = [key for key in keys if key not in row]
    if missing:
        raise CatalogBuildError(f"Catalog record is missing required fields: {', '.join(missing)}")
    return tuple(row[key] for key in keys)


def _rows(records: Iterable[CatalogRow], kind: str) -> list[CatalogRow]:
    result = list(records)
    if any(not isinstance(row, Mapping) for row in result):
        raise CatalogBuildError(f"Every {kind} record must be a mapping")
    return result


def _flag(row: CatalogRow, key: str, default: int) -> int:
    value = row.get(key, default)
    if value not in (0, 1, False, True):
        raise CatalogBuildError(f"{key} must be 0 or 1")
    return int(value)


def build_catalog(
    output_path: Path | str,
    *,
    documents: Iterable[CatalogRow],
    source_files: Iterable[CatalogRow],
    chunks: Iterable[CatalogRow],
    relationships: Iterable[CatalogRow] = (),
    scope_texts: Iterable[CatalogRow] = (),
    operator_overrides: Iterable[CatalogRow] = (),
    missing_volumes: Iterable[CatalogRow] = (),
) -> Path:
    """Create a new catalog database from validated, already-extracted records.

    The target must not exist. That makes this safe to use with immutable release
    directories and prevents an accidental rebuild from replacing a live catalog.
    """
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing catalog: {target}")

    document_rows = _rows(documents, "document")
    source_rows = _rows(source_files, "source file")
    chunk_rows = _rows(chunks, "chunk")
    relationship_rows = _rows(relationships, "relationship")
    scope_rows = _rows(scope_texts, "scope text")
    override_rows = _rows(operator_overrides, "operator override")
    missing_rows = _rows(missing_volumes, "missing volume")
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        connection = sqlite3.connect(temporary_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)

        document_by_id: dict[Any, CatalogRow] = {}
        for row in document_rows:
            (doc_id,) = _required(row, "id")
            code = row.get("code")
            title = row.get("title")
            document_by_id[doc_id] = row
            eligible = _flag(row, "is_eligible", 1)
            is_stub = _flag(row, "is_stub", 0)
            if is_stub and eligible:
                raise CatalogBuildError(f"Standard stub {code!r} cannot be recommendation-eligible")
            connection.execute(
                """INSERT INTO logical_documents
                (id, family, number, part, year, code, title, status, edition,
                 identity_source, identity_status, is_eligible, is_stub, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    row.get("family"),
                    row.get("number"),
                    row.get("part"),
                    row.get("year"),
                    code,
                    title,
                    row.get("status"),
                    row.get("edition"),
                    row.get("identity_source"),
                    row.get("identity_status", "verified"),
                    eligible,
                    is_stub,
                    row.get("page_count", 0),
                ),
            )

        for row in source_rows:
            source_id, doc_id = _required(row, "id", "logical_doc_id")
            document = document_by_id.get(doc_id)
            if document is None:
                raise CatalogBuildError(
                    f"Source file {source_id!r} references missing document {doc_id!r}"
                )
            if _flag(document, "is_stub", 0):
                raise CatalogBuildError(
                    f"Standard stub {doc_id!r} cannot have ingested source files"
                )
            connection.execute(
                """INSERT INTO source_files
                (id, logical_doc_id, file_name, relative_path, sha256, page_count,
                 part_index, page_offset, extraction_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_id,
                    doc_id,
                    row.get("file_name"),
                    row.get("relative_path"),
                    row.get("sha256"),
                    row.get("page_count", 0),
                    row.get("part_index"),
                    row.get("page_offset", 0),
                    row.get("extraction_status", "processed"),
                ),
            )

        source_by_id = {row["id"]: row for row in source_rows}

        for row in override_rows:
            override_id, doc_id, source_id, asserted_identity, operator_id, timestamp, reason = _required(
                row, "override_id", "logical_doc_id", "source_file_id", "asserted_identity",
                "operator_id", "timestamp", "reason"
            )
            source = source_by_id.get(source_id)
            if source is None or source.get("logical_doc_id") != doc_id:
                raise CatalogBuildError(
                    f"Operator override {override_id!r} must reference a source file in its logical document"
                )
            evidence_sources = row.get("evidence_sources")
            if not isinstance(evidence_sources, list) or not evidence_sources or any(
                not isinstance(item, str) or not item.strip() for item in evidence_sources
            ):
                raise CatalogBuildError(
                    f"Operator override {override_id!r} requires evidence_sources"
                )
            connection.execute(
                """INSERT INTO operator_overrides
                (override_id, logical_doc_id, source_file_id, asserted_identity,
                 operator_id, timestamp, reason, evidence_sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (override_id, doc_id, source_id, asserted_identity, operator_id, timestamp,
                 reason, json.dumps(evidence_sources, ensure_ascii=False)),
            )

        for row in missing_rows:
            missing_id, doc_id, volume_id, page_start, page_end = _required(
                row, "missing_volume_id", "logical_doc_id", "volume_id",
                "absolute_page_start", "absolute_page_end"
            )
            if doc_id not in document_by_id or document_by_id[doc_id].get("is_stub"):
                raise CatalogBuildError(
                    f"Missing volume {missing_id!r} must reference an ingested logical document"
                )
            connection.execute(
                """INSERT INTO missing_volumes
                (missing_volume_id, logical_doc_id, volume_id, absolute_page_start,
                 absolute_page_end, reason) VALUES (?, ?, ?, ?, ?, ?)""",
                (missing_id, doc_id, volume_id, page_start, page_end, row.get("reason")),
            )

        for row in relationship_rows:
            source_code, target_code, relationship_type = _required(
                row, "source_code", "target_code", "relationship_type"
            )
            connection.execute(
                """INSERT INTO relationships
                (source_code, target_code, relationship_type, page_number, evidence_text)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    source_code,
                    target_code,
                    relationship_type,
                    row.get("page_number"),
                    row.get("evidence_text", ""),
                ),
            )

        for row in scope_rows:
            (standard_code,) = _required(row, "standard_code")
            connection.execute(
                "INSERT INTO scope_text (standard_code, scope, exclusions) VALUES (?, ?, ?)",
                (standard_code, row.get("scope"), row.get("exclusions")),
            )

        for row in chunk_rows:
            chunk_id, doc_id, standard_code, source_id = _required(
                row, "chunk_id", "logical_doc_id", "standard_code", "source_file_id"
            )
            page_start, page_end, absolute_start, absolute_end, text = _required(
                row, "page_start", "page_end", "absolute_page_start", "absolute_page_end", "text"
            )
            if page_start < 1 or page_end < page_start:
                raise CatalogBuildError(f"Invalid physical page range for chunk {chunk_id!r}")
            if absolute_start < 1 or absolute_end < absolute_start:
                raise CatalogBuildError(f"Invalid absolute page range for chunk {chunk_id!r}")
            source = source_by_id.get(source_id)
            if source is None:
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} references missing source file {source_id!r}"
                )
            if source.get("logical_doc_id") != doc_id:
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} and source file {source_id!r} belong to different documents"
                )
            document = document_by_id.get(doc_id)
            if document is None or document.get("code") != standard_code:
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} standard code does not match its logical document"
                )
            chunk_eligible = _flag(row, "is_eligible", 1)
            if chunk_eligible and (
                _flag(document, "is_stub", 0) or not _flag(document, "is_eligible", 1)
            ):
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} cannot be eligible when its document is ineligible"
                )
            physical_count = source.get("page_count", 0)
            offset = source.get("page_offset", 0)
            if page_end > physical_count:
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} physical page range exceeds its source file"
                )
            if absolute_start != offset + page_start or absolute_end != offset + page_end:
                raise CatalogBuildError(
                    f"Chunk {chunk_id!r} absolute pages do not match source offset mapping"
                )
            connection.execute(
                """INSERT INTO chunks
                (chunk_id, logical_doc_id, standard_code, source_file_id, page_start,
                 page_end, absolute_page_start, absolute_page_end, section_number,
                 section_title, text, is_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    doc_id,
                    standard_code,
                    source_id,
                    page_start,
                    page_end,
                    absolute_start,
                    absolute_end,
                    row.get("section_number"),
                    row.get("section_title"),
                    text,
                    chunk_eligible,
                ),
            )
            connection.execute(
                """INSERT INTO chunks_fts
                (chunk_id, standard_code, section_number, section_title, text)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    standard_code,
                    row.get("section_number") or "",
                    row.get("section_title") or "",
                    text,
                ),
            )

        connection.commit()
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise CatalogBuildError(f"SQLite integrity check failed: {integrity!r}")
        connection.close()
        connection = None
        # A hard-link promotion is atomic and fails if another builder created
        # the immutable target after our initial existence check.
        try:
            os.link(temporary_path, target)
        except (OSError, PermissionError):
            if target.exists():
                raise FileExistsError(f"Refusing to replace existing catalog: {target}")
            os.replace(temporary_path, target)
        if temporary_path.exists():
            temporary_path.unlink()
        temporary_path = None
        return target
    except (sqlite3.Error, TypeError) as exc:
        raise CatalogBuildError(f"Could not build SQLite catalog: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_catalog_payloads(
    catalog_path: Path | str,
    chunks_path: Path | str,
    *,
    documents: Iterable[CatalogRow],
    source_files: Iterable[CatalogRow],
    chunks: Iterable[CatalogRow],
    relationships: Iterable[CatalogRow] = (),
    scope_texts: Iterable[CatalogRow] = (),
    operator_overrides: Iterable[CatalogRow] = (),
    missing_volumes: Iterable[CatalogRow] = (),
) -> tuple[Path, Path]:
    """Build `catalog.sqlite` and its ordered `chunks.jsonl` release payload."""
    catalog_target = Path(catalog_path)
    chunks_target = Path(chunks_path)
    if catalog_target.resolve() == chunks_target.resolve():
        raise ValueError("catalog_path and chunks_path must be different files")
    if catalog_target.exists() or chunks_target.exists():
        raise FileExistsError("Refusing to replace existing catalog release payloads")

    document_rows = _rows(documents, "document")
    source_rows = _rows(source_files, "source file")
    chunk_rows = _rows(chunks, "chunk")
    relationship_rows = _rows(relationships, "relationship")
    scope_rows = _rows(scope_texts, "scope text")
    override_rows = _rows(operator_overrides, "operator override")
    missing_rows = _rows(missing_volumes, "missing volume")
    chunks_target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    published_chunks = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{chunks_target.name}.",
            suffix=".tmp",
            dir=chunks_target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in chunk_rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, chunks_target)
        except (OSError, PermissionError):
            if chunks_target.exists():
                raise FileExistsError(f"Refusing to replace existing chunks file: {chunks_target}")
            os.replace(temporary_path, chunks_target)
        published_chunks = True
        if temporary_path.exists():
            temporary_path.unlink()
        temporary_path = None
        build_catalog(
            catalog_target,
            documents=document_rows,
            source_files=source_rows,
            chunks=chunk_rows,
            relationships=relationship_rows,
            scope_texts=scope_rows,
            operator_overrides=override_rows,
            missing_volumes=missing_rows,
        )
        return catalog_target, chunks_target
    except Exception:
        if published_chunks:
            chunks_target.unlink(missing_ok=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
