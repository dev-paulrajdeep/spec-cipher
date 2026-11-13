"""Write a checksum manifest for a fully assembled staged release."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from speccipher.release.manifest import REQUIRED_PAYLOADS, SCHEMA_VERSION


class ManifestBuildError(ValueError):
    """Raised when staged payloads cannot produce a consistent release manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _counts(release_dir: Path) -> dict[str, int]:
    catalog_path = release_dir / "catalog.sqlite"
    try:
        connection = sqlite3.connect(f"{catalog_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        try:
            documents = connection.execute("SELECT COUNT(*) FROM logical_documents").fetchone()[0]
            eligible_documents = connection.execute(
                "SELECT COUNT(*) FROM logical_documents WHERE is_eligible = 1 AND is_stub = 0"
            ).fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            eligible_chunks = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE is_eligible = 1"
            ).fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ManifestBuildError(f"Could not read counts from catalog.sqlite: {exc}") from exc

    try:
        chunk_ids = json.loads((release_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Could not read chunk_ids.json: {exc}") from exc
    if not isinstance(chunk_ids, list) or any(
        not isinstance(item, str) or not item for item in chunk_ids
    ):
        raise ManifestBuildError("chunk_ids.json must be an array of non-empty strings")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ManifestBuildError("chunk_ids.json contains duplicate IDs")
    if chunks != len(chunk_ids):
        raise ManifestBuildError(
            f"Catalog chunk count {chunks} does not match chunk_ids.json length {len(chunk_ids)}"
        )
    return {
        "documents": documents,
        "eligible_documents": eligible_documents,
        "chunks": chunks,
        "eligible_chunks": eligible_chunks,
        "vectors": len(chunk_ids),
    }


def write_release_manifest(
    release_dir: Path | str,
    *,
    embedding_model: str,
    embedding_revision: str,
    git_sha: str | None,
    chunking_version: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Derive and atomically write `release.json` for staged payload files.

    All required payloads must already exist. The destination manifest cannot be
    replaced, so a staged release is not silently rewritten in place.
    """
    directory = Path(release_dir)
    release_id = directory.name
    if directory.is_symlink() or not directory.is_dir():
        raise ManifestBuildError("Staging release must be a real directory")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id) is None:
        raise ManifestBuildError(f"Invalid release ID: {release_id!r}")
    manifest_path = directory / "release.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace release manifest: {manifest_path}")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (embedding_model, embedding_revision, chunking_version)
    ):
        raise ManifestBuildError(
            "Embedding model, embedding revision, and chunking version are required"
        )
    if git_sha is not None and not isinstance(git_sha, str):
        raise ManifestBuildError("git_sha must be a string or None")

    missing = [
        name
        for name in sorted(REQUIRED_PAYLOADS)
        if not (directory / name).is_file() or (directory / name).is_symlink()
    ]
    if missing:
        raise ManifestBuildError(f"Missing required release payloads: {', '.join(missing)}")

    try:
        quality = json.loads((directory / "QUALITY.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Could not read QUALITY.json: {exc}") from exc
    if not isinstance(quality, Mapping):
        raise ManifestBuildError("QUALITY.json must contain a JSON object")

    counts = _counts(directory)
    if quality.get("release_id") != release_id:
        raise ManifestBuildError("QUALITY.json release_id does not match staging directory")
    for quality_key, count_key in (("total_documents", "documents"), ("total_chunks", "chunks")):
        if quality_key in quality and quality[quality_key] != counts[count_key]:
            raise ManifestBuildError(f"QUALITY.json {quality_key} does not match catalog")

    timestamp = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if created_at is None
        else created_at
    )
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ManifestBuildError("created_at must be an ISO-8601 timestamp") from exc
    if parsed_timestamp.tzinfo is None:
        raise ManifestBuildError("created_at must include a timezone")
    files = {
        filename: {
            "sha256": f"sha256:{_sha256(directory / filename)}",
            "bytes": (directory / filename).stat().st_size,
        }
        for filename in sorted(REQUIRED_PAYLOADS)
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "created_at": timestamp,
        "git_sha": git_sha,
        "chunking_version": chunking_version,
        "embedding": {
            "model": embedding_model,
            "revision": embedding_revision,
            "dimension": 1024,
            "distance_metric": "cosine",
            "normalized": True,
        },
        "index_type": "IndexFlatIP",
        "counts": counts,
        "files": files,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".release.",
            suffix=".tmp",
            dir=directory,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create-if-absent protects staged manifests from concurrent writers.
        try:
            os.link(temporary_path, manifest_path)
        except (OSError, PermissionError):
            if manifest_path.exists():
                raise FileExistsError(f"Refusing to replace existing manifest: {manifest_path}")
            os.replace(temporary_path, manifest_path)
        if temporary_path.exists():
            temporary_path.unlink()
        temporary_path = None
    except OSError as exc:
        raise ManifestBuildError(f"Could not atomically write release manifest: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest
