"""Corpus Release manifest verification and CURRENT pointer resolution."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "2.0"
REQUIRED_PAYLOADS = frozenset(
    {"catalog.sqlite", "chunks.jsonl", "vectors.faiss", "chunk_ids.json", "QUALITY.json"}
)


class ReleaseIntegrityError(Exception):
    """Raised when corpus release verification fails."""


class ReleaseNotFoundError(FileNotFoundError):
    """Raised when release directory or pointer target is not found."""


def resolve_release_path(release_path: Path | str) -> Path:
    """Resolve a release directory or digest-checked CURRENT pointer."""
    path = Path(release_path)
    if not path.is_file():
        return path
    if path.name != "CURRENT":
        raise ReleaseIntegrityError("Expected a release directory or a CURRENT pointer file")

    try:
        target_name = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ReleaseIntegrityError(f"Failed to read release pointer file '{path}': {exc}") from exc

    expected_manifest_digest: str | None = None
    has_manifest_digest = False
    try:
        pointer = json.loads(target_name)
    except json.JSONDecodeError:
        # Continue to read legacy one-line pointers during the transition. New
        # publishers always write the digest-bearing JSON form.
        release_id = target_name
    else:
        if not isinstance(pointer, dict) or set(pointer) != {"release_id", "release_sha256"}:
            raise ReleaseIntegrityError("CURRENT must contain release_id and release_sha256")
        release_id = pointer["release_id"]
        expected_manifest_digest = pointer["release_sha256"]
        has_manifest_digest = True

    if (
        not isinstance(release_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id) is None
    ):
        raise ReleaseIntegrityError(
            f"CURRENT pointer must contain a safe release id, got {release_id!r}"
        )
    if has_manifest_digest:
        if not isinstance(expected_manifest_digest, str):
            raise ReleaseIntegrityError("CURRENT release_sha256 must be a SHA-256 digest")
        expected_manifest_digest = expected_manifest_digest.removeprefix("sha256:").lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_digest) is None:
            raise ReleaseIntegrityError("CURRENT release_sha256 must be a SHA-256 digest")

    releases_root = path.parent.resolve()
    target_dir = (path.parent / release_id).resolve()
    if target_dir.parent != releases_root:
        raise ReleaseIntegrityError(
            "CURRENT pointer target must remain inside the releases directory"
        )
    if not target_dir.is_dir():
        raise ReleaseNotFoundError(
            f"CURRENT pointer '{path}' points to missing release '{release_id}'"
        )
    if expected_manifest_digest is not None:
        actual_digest = _file_digest(target_dir / "release.json")
        if actual_digest != expected_manifest_digest:
            raise ReleaseIntegrityError("CURRENT release_sha256 does not match release.json")
    return target_dir


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseIntegrityError(f"Invalid {description} JSON at '{path}': {exc}") from exc


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ReleaseIntegrityError(f"Could not read release payload '{path.name}': {exc}") from exc
    return digest.hexdigest()


def _payload_paths(release_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ReleaseIntegrityError("Manifest field 'files' must be an object of payload checksums")

    missing_checksums = REQUIRED_PAYLOADS - files.keys()
    if missing_checksums:
        names = ", ".join(sorted(missing_checksums))
        raise ReleaseIntegrityError(f"Missing checksum entries for required payloads: {names}")

    paths: dict[str, Path] = {}
    for filename, metadata in files.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or PureWindowsPath(filename).name != filename
        ):
            raise ReleaseIntegrityError(f"Unsafe payload filename in manifest: {filename!r}")
        payload_path = release_dir / filename
        if not payload_path.is_file():
            raise ReleaseIntegrityError(f"Required constituent file missing: {filename}")

        digest_value = metadata.get("sha256") if isinstance(metadata, dict) else metadata
        if not isinstance(digest_value, str):
            raise ReleaseIntegrityError(f"Missing SHA-256 checksum for payload '{filename}'")
        expected_digest = digest_value.removeprefix("sha256:").lower()
        if len(expected_digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in expected_digest
        ):
            raise ReleaseIntegrityError(f"Invalid SHA-256 checksum for payload '{filename}'")

        actual_digest = _file_digest(payload_path)
        if actual_digest != expected_digest:
            raise ReleaseIntegrityError(
                f"SHA-256 mismatch for {filename}: expected {expected_digest}, got {actual_digest}"
            )
        if isinstance(metadata, dict) and "bytes" in metadata:
            expected_size = metadata["bytes"]
            if not isinstance(expected_size, int) or expected_size != payload_path.stat().st_size:
                raise ReleaseIntegrityError(f"Byte count mismatch for payload '{filename}'")
        paths[filename] = payload_path
    return paths


def _verify_catalog(
    catalog_path: Path,
    manifest: dict[str, Any],
    expected_chunk_ids: list[str],
    *,
    check_fts_integrity: bool,
) -> None:
    try:
        mode = "rw" if check_fts_integrity else "ro"
        connection = sqlite3.connect(f"{catalog_path.resolve().as_uri()}?mode={mode}", uri=True)
    except sqlite3.Error as exc:
        raise ReleaseIntegrityError(f"Could not open catalog read-only: {exc}") from exc

    try:
        if not check_fts_integrity:
            connection.execute("PRAGMA query_only = ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ReleaseIntegrityError(f"SQLite catalog integrity check failed: {integrity}")

        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        required_tables = {
            "logical_documents",
            "source_files",
            "chunks",
            "chunks_fts",
            "relationships",
            "scope_text",
        }
        missing_tables = required_tables - table_names
        if missing_tables:
            raise ReleaseIntegrityError(
                f"Catalog is missing required tables: {', '.join(sorted(missing_tables))}"
            )

        counts = manifest["counts"]
        document_count = connection.execute("SELECT COUNT(*) FROM logical_documents").fetchone()[0]
        eligible_documents = connection.execute(
            "SELECT COUNT(*) FROM logical_documents WHERE is_eligible = 1 AND is_stub = 0"
        ).fetchone()[0]
        chunk_rows = connection.execute(
            "SELECT chunk_id, logical_doc_id, source_file_id FROM chunks ORDER BY chunk_id"
        ).fetchall()
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

        if document_count != counts["documents"]:
            raise ReleaseIntegrityError(
                f"Catalog document count {document_count} does not match manifest count {counts['documents']}"
            )
        if eligible_documents != counts["eligible_documents"]:
            raise ReleaseIntegrityError("Catalog eligible document count does not match manifest")
        catalog_chunk_ids = [row[0] for row in chunk_rows]
        if len(catalog_chunk_ids) != counts["chunks"]:
            raise ReleaseIntegrityError("Catalog chunk count does not match manifest")
        if sorted(expected_chunk_ids) != catalog_chunk_ids:
            raise ReleaseIntegrityError("chunk_ids.json does not match catalog chunk IDs")
        if fts_count != counts["chunks"]:
            raise ReleaseIntegrityError("Catalog FTS row count does not match catalog chunk count")

        if check_fts_integrity:
            connection.execute("BEGIN")
            connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")
            connection.rollback()

        orphan_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks AS c
            LEFT JOIN logical_documents AS d ON d.id = c.logical_doc_id
            LEFT JOIN source_files AS s ON s.id = c.source_file_id
            WHERE d.id IS NULL OR s.id IS NULL
            """
        ).fetchone()[0]
        if orphan_count:
            raise ReleaseIntegrityError(
                f"Catalog has {orphan_count} chunks with missing provenance"
            )

        eligible_chunks = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE is_eligible = 1"
        ).fetchone()[0]
        if eligible_chunks != counts["eligible_chunks"]:
            raise ReleaseIntegrityError("Catalog eligible chunk count does not match manifest")
    except sqlite3.Error as exc:
        raise ReleaseIntegrityError(f"Catalog schema or query validation failed: {exc}") from exc
    finally:
        connection.close()


def _verify_chunks(chunks_path: Path, ids_path: Path, manifest: dict[str, Any]) -> list[str]:
    chunk_ids = _read_json(ids_path, "chunk_ids")
    if not isinstance(chunk_ids, list) or any(
        not isinstance(item, str) or not item for item in chunk_ids
    ):
        raise ReleaseIntegrityError("chunk_ids.json must be an array of non-empty strings")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ReleaseIntegrityError("chunk_ids.json contains duplicate chunk IDs")

    chunk_rows: dict[str, dict[str, Any]] = {}
    try:
        with chunks_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReleaseIntegrityError(
                        f"Invalid chunk JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
                    raise ReleaseIntegrityError(f"Chunk line {line_number} has no valid chunk_id")
                chunk_id = chunk["chunk_id"]
                if chunk_id in chunk_rows:
                    raise ReleaseIntegrityError(f"Duplicate chunk ID in chunks.jsonl: {chunk_id}")
                chunk_rows[chunk_id] = chunk
    except OSError as exc:
        raise ReleaseIntegrityError(f"Could not read chunks.jsonl: {exc}") from exc

    if len(chunk_rows) != manifest["counts"]["chunks"]:
        raise ReleaseIntegrityError("chunks.jsonl count does not match manifest")
    if set(chunk_rows) != set(chunk_ids):
        raise ReleaseIntegrityError("chunk_ids.json does not match chunks.jsonl chunk IDs")
    return chunk_ids


def verify_release(
    release_dir: Path | str,
    *,
    check_fts_integrity: bool = False,
) -> dict[str, Any]:
    """Verify manifest, payload digests, chunk identity and catalog integrity.

    `check_fts_integrity` is for an unpublished staging release. It runs SQLite's
    FTS5 integrity command in a rolled-back transaction before promotion. Serving
    verification stays read-only and relies on payload digests from publication.
    """
    resolved_dir = resolve_release_path(release_dir)
    if not resolved_dir.is_dir():
        raise ReleaseNotFoundError(f"Release directory does not exist: {resolved_dir}")

    manifest_path = resolved_dir / "release.json"
    if not manifest_path.is_file():
        raise ReleaseIntegrityError(f"Missing release manifest: {manifest_path}")
    manifest = _read_json(manifest_path, "release manifest")
    if not isinstance(manifest, dict):
        raise ReleaseIntegrityError("Release manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseIntegrityError(
            f"Unsupported release schema version: {manifest.get('schema_version')!r}"
        )
    if manifest.get("release_id") != resolved_dir.name:
        raise ReleaseIntegrityError("Manifest release_id does not match release directory name")
    release_id = manifest["release_id"]
    if (
        not isinstance(release_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id) is None
        or PureWindowsPath(release_id).drive
    ):
        raise ReleaseIntegrityError(f"Invalid release ID: {release_id!r}")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise ReleaseIntegrityError("Manifest created_at must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseIntegrityError("Manifest created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ReleaseIntegrityError("Manifest created_at must include a timezone")

    counts = manifest.get("counts")
    required_counts = {"documents", "eligible_documents", "chunks", "eligible_chunks", "vectors"}
    if not isinstance(counts, dict) or required_counts - counts.keys():
        raise ReleaseIntegrityError("Manifest counts are missing required fields")
    if any(
        not isinstance(counts[key], int) or isinstance(counts[key], bool) or counts[key] < 0
        for key in required_counts
    ):
        raise ReleaseIntegrityError("Manifest counts must be non-negative integers")
    if counts["eligible_documents"] > counts["documents"]:
        raise ReleaseIntegrityError("Eligible document count exceeds total documents")
    if counts["eligible_chunks"] > counts["chunks"]:
        raise ReleaseIntegrityError("Eligible chunk count exceeds total chunks")
    if counts["chunks"] != counts["vectors"]:
        raise ReleaseIntegrityError(
            f"Counts mismatch: chunks={counts['chunks']} vs vectors={counts['vectors']}"
        )

    embedding = manifest.get("embedding")
    dimension = (
        embedding.get("dim", embedding.get("dimension")) if isinstance(embedding, dict) else None
    )
    if not isinstance(dimension, int) or dimension != 1024:
        raise ReleaseIntegrityError(f"Release embedding dimension must be 1024, got {dimension!r}")
    if not isinstance(embedding.get("model"), str) or not embedding["model"].strip():
        raise ReleaseIntegrityError("Manifest embedding model must be a non-empty string")
    if not isinstance(embedding.get("revision"), str) or not embedding["revision"].strip():
        raise ReleaseIntegrityError("Manifest embedding revision must be a non-empty string")
    if embedding.get("distance_metric") != "cosine":
        raise ReleaseIntegrityError("Manifest embedding distance_metric must be cosine")
    if embedding.get("normalized") is not True:
        raise ReleaseIntegrityError("Manifest embedding vectors must be normalized")
    if manifest.get("index_type") != "IndexFlatIP":
        raise ReleaseIntegrityError("Release index_type must be IndexFlatIP")

    payloads = _payload_paths(resolved_dir, manifest)
    chunk_ids = _verify_chunks(payloads["chunks.jsonl"], payloads["chunk_ids.json"], manifest)
    if len(chunk_ids) != counts["vectors"]:
        raise ReleaseIntegrityError("Vector count does not match chunk_ids.json length")
    _verify_catalog(
        payloads["catalog.sqlite"],
        manifest,
        chunk_ids,
        check_fts_integrity=check_fts_integrity,
    )

    quality = _read_json(payloads["QUALITY.json"], "quality report")
    if not isinstance(quality, dict) or quality.get("release_id") != release_id:
        raise ReleaseIntegrityError("QUALITY.json release_id does not match manifest")
    for quality_key, count_key in (("total_documents", "documents"), ("total_chunks", "chunks")):
        if quality_key in quality and quality[quality_key] != counts[count_key]:
            raise ReleaseIntegrityError(f"QUALITY.json {quality_key} does not match manifest")

    vectors_path = payloads["vectors.faiss"]
    try:
        import faiss

        index = faiss.read_index(str(vectors_path))
    except Exception as exc:
        raise ReleaseIntegrityError(f"Failed to inspect FAISS index: {exc}") from exc
    if type(index).__name__ != "IndexFlatIP":
        raise ReleaseIntegrityError(f"FAISS index must be IndexFlatIP, got {type(index).__name__}")
    if index.d != dimension:
        raise ReleaseIntegrityError(
            f"Vector dimension mismatch: FAISS index dimension {index.d} != manifest dimension {dimension}"
        )
    if index.ntotal != counts["vectors"]:
        raise ReleaseIntegrityError("FAISS vector count does not match manifest")

    return manifest
