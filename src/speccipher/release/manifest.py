"""Corpus Release manifest verification and CURRENT pointer resolution (Layer 3 Release)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ReleaseIntegrityError(Exception):
    """Raised when corpus release verification fails."""


class ReleaseNotFoundError(FileNotFoundError):
    """Raised when release directory or pointer target is not found."""


def resolve_release_path(release_path: Path | str) -> Path:
    """Resolve release directory, dereferencing CURRENT text pointer file if necessary."""
    p = Path(release_path)
    if p.is_file():
        try:
            target_name = p.read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise ReleaseIntegrityError(f"Failed to read release pointer file '{p}': {exc}") from exc

        if not target_name:
            raise ReleaseIntegrityError(f"Release pointer file '{p}' is empty")

        target_dir = p.parent / target_name
        if not target_dir.exists() or not target_dir.is_dir():
            target_dir = Path(target_name)

        if not target_dir.exists() or not target_dir.is_dir():
            raise ReleaseNotFoundError(
                f"Release pointer file '{p}' points to non-existent directory '{target_name}'"
            )
        return target_dir
    return p


def verify_release(release_dir: Path | str) -> dict[str, Any]:
    """Verify release integrity fail-closed."""
    resolved_dir = resolve_release_path(release_dir)
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        raise ReleaseNotFoundError(f"Release directory does not exist: {resolved_dir}")

    manifest_path = resolved_dir / "release.json"
    if not manifest_path.exists():
        raise ReleaseIntegrityError(f"Missing release manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReleaseIntegrityError(f"Corrupt manifest JSON: {e}") from e

    # Check required constituent files
    for fname in ["catalog.sqlite", "chunks.jsonl", "vectors.faiss", "chunk_ids.json"]:
        fpath = resolved_dir / fname
        if not fpath.exists():
            raise ReleaseIntegrityError(f"Required constituent file missing: {fname}")

    # Verify checksums in files map
    files_map = manifest.get("files", {})
    for fname, meta in files_map.items():
        fpath = resolved_dir / fname
        if not fpath.exists():
            raise ReleaseIntegrityError(f"File listed in manifest missing: {fname}")
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest()
        expected_hash = meta.get("sha256")
        if actual_hash != expected_hash:
            raise ReleaseIntegrityError(
                f"SHA-256 mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
            )

    # Verify counts alignment
    counts = manifest.get("counts", {})
    if counts.get("chunks") != counts.get("vectors"):
        raise ReleaseIntegrityError(
            f"Counts mismatch: chunks={counts.get('chunks')} vs vectors={counts.get('vectors')}"
        )

    # Verify vector index dimension if faiss is available and index exists
    manifest_dim = manifest.get("embedding", {}).get("dim")
    if manifest_dim is not None:
        vectors_path = resolved_dir / "vectors.faiss"
        if vectors_path.exists():
            try:
                import faiss

                idx = faiss.read_index(str(vectors_path))
                if idx.d != manifest_dim:
                    raise ReleaseIntegrityError(
                        f"Vector dimension mismatch: FAISS index dimension {idx.d} != manifest dimension {manifest_dim}"
                    )
            except ImportError:
                pass
            except ReleaseIntegrityError:
                raise
            except Exception as exc:
                raise ReleaseIntegrityError(f"Failed to inspect FAISS index dimension: {exc}") from exc

    return manifest
