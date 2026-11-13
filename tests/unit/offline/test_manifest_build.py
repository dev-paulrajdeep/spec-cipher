from __future__ import annotations

import hashlib
import json

import pytest

from speccipher.offline.indexing.fts_build import build_catalog
from speccipher.offline.publishing.manifest import ManifestBuildError, write_release_manifest


def _staged_release(tmp_path):
    release_dir = tmp_path / "20261112T100000Z-test01"
    release_dir.mkdir()
    build_catalog(
        release_dir / "catalog.sqlite",
        documents=[],
        source_files=[],
        chunks=[],
    )
    (release_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    (release_dir / "chunk_ids.json").write_text("[]\n", encoding="utf-8")
    (release_dir / "vectors.faiss").write_bytes(b"fake index payload")
    (release_dir / "QUALITY.json").write_text(
        json.dumps({"release_id": release_dir.name, "total_documents": 0, "total_chunks": 0}),
        encoding="utf-8",
    )
    return release_dir


def test_manifest_uses_catalog_counts_and_payload_checksums(tmp_path):
    release_dir = _staged_release(tmp_path)

    manifest = write_release_manifest(
        release_dir,
        embedding_model="BAAI/bge-m3",
        embedding_revision="main",
        git_sha="abc1234",
        chunking_version="2.0",
        created_at="2026-11-12T10:00:00Z",
    )

    assert manifest["counts"] == {
        "documents": 0,
        "eligible_documents": 0,
        "chunks": 0,
        "eligible_chunks": 0,
        "vectors": 0,
    }
    assert manifest["embedding"]["dimension"] == 1024
    assert manifest["files"]["vectors.faiss"]["sha256"] == (
        "sha256:" + hashlib.sha256(b"fake index payload").hexdigest()
    )
    assert json.loads((release_dir / "release.json").read_text(encoding="utf-8")) == manifest


def test_manifest_rejects_chunk_id_count_mismatch(tmp_path):
    release_dir = _staged_release(tmp_path)
    (release_dir / "chunk_ids.json").write_text('["orphan"]\n', encoding="utf-8")

    with pytest.raises(ManifestBuildError, match="does not match chunk_ids.json"):
        write_release_manifest(
            release_dir,
            embedding_model="BAAI/bge-m3",
            embedding_revision="main",
            git_sha=None,
            chunking_version="2.0",
        )
    assert not (release_dir / "release.json").exists()


def test_manifest_refuses_replacing_existing_manifest(tmp_path):
    release_dir = _staged_release(tmp_path)
    path = release_dir / "release.json"
    path.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_release_manifest(
            release_dir,
            embedding_model="BAAI/bge-m3",
            embedding_revision="main",
            git_sha=None,
            chunking_version="2.0",
        )
    assert path.read_text(encoding="utf-8") == "keep"
