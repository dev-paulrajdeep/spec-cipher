"""Unit tests for fail-closed Corpus Release validation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from speccipher.config.settings import Settings
from speccipher.release.manifest import ReleaseIntegrityError, resolve_release_path, verify_release
from speccipher.release.publisher import ReleasePublisher, ReleasePublishError
from speccipher.services.container import ApplicationContainer


@pytest.fixture(autouse=True)
def fake_faiss_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep release contract tests independent of the local FAISS DLL load."""

    class IndexFlatIP:
        d = 1024
        ntotal = 1

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(read_index=lambda _path: IndexFlatIP()),
    )


def _write_release(release_dir: Path) -> Path:
    release_dir.mkdir(parents=True)
    catalog_path = release_dir / "catalog.sqlite"
    conn = sqlite3.connect(catalog_path)
    try:
        conn.executescript(
            """
            CREATE TABLE logical_documents (
                id TEXT PRIMARY KEY, code TEXT UNIQUE, is_eligible INTEGER DEFAULT 1, is_stub INTEGER DEFAULT 0
            );
            CREATE TABLE source_files (id TEXT PRIMARY KEY, logical_doc_id TEXT);
            CREATE TABLE relationships (
                source_code TEXT, target_code TEXT, relationship_type TEXT, page_number INTEGER,
                evidence_text TEXT
            );
            CREATE TABLE scope_text (standard_code TEXT PRIMARY KEY, scope TEXT, exclusions TEXT);
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY, logical_doc_id TEXT, source_file_id TEXT,
                absolute_page_start INTEGER, absolute_page_end INTEGER, text TEXT,
                is_eligible INTEGER DEFAULT 1
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);
            INSERT INTO logical_documents (id, code) VALUES ('doc-1', 'IS 1:2020');
            INSERT INTO source_files (id, logical_doc_id) VALUES ('source-1', 'doc-1');
            INSERT INTO chunks VALUES ('chunk-1', 'doc-1', 'source-1', 1, 1, 'specification text', 1);
            INSERT INTO chunks_fts VALUES ('chunk-1', 'specification text');
            """
        )
    finally:
        conn.close()

    chunk: dict[str, Any] = {
        "chunk_id": "chunk-1",
        "logical_doc_id": "doc-1",
        "source_file_id": "source-1",
        "absolute_page_start": 1,
        "absolute_page_end": 1,
        "text": "specification text",
    }
    payloads = {
        "chunks.jsonl": (json.dumps(chunk) + "\n").encode("utf-8"),
        "chunk_ids.json": json.dumps(["chunk-1"]).encode("utf-8"),
        "vectors.faiss": b"test-vector-index-payload",
        "QUALITY.json": json.dumps({"release_id": release_dir.name}).encode("utf-8"),
    }
    for filename, content in payloads.items():
        (release_dir / filename).write_bytes(content)

    required_files = ["catalog.sqlite", *payloads]
    file_metadata = {}
    for filename in required_files:
        content = (release_dir / filename).read_bytes()
        file_metadata[filename] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    manifest = {
        "schema_version": "2.0",
        "release_id": release_dir.name,
        "created_at": "2026-11-12T00:00:00Z",
        "chunking_version": "1.0",
        "embedding": {
            "model": "synthetic-test",
            "revision": "test",
            "dim": 1024,
            "distance_metric": "cosine",
            "normalized": True,
        },
        "index_type": "IndexFlatIP",
        "counts": {
            "documents": 1,
            "eligible_documents": 1,
            "chunks": 1,
            "eligible_chunks": 1,
            "vectors": 1,
        },
        "files": file_metadata,
    }
    (release_dir / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release_dir


def _read_manifest(release_dir: Path) -> dict[str, Any]:
    return json.loads((release_dir / "release.json").read_text(encoding="utf-8"))


def _write_manifest(release_dir: Path, manifest: dict[str, Any]) -> None:
    (release_dir / "release.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_release_accepts_consistent_synthetic_release(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")

    manifest = verify_release(release_dir)

    assert manifest["release_id"] == "release-1"


def test_verify_release_requires_checksum_for_every_payload(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "release-1")
    manifest = _read_manifest(release_dir)
    del manifest["files"]["QUALITY.json"]
    _write_manifest(release_dir, manifest)

    with pytest.raises(
        ReleaseIntegrityError, match="checksum.*QUALITY.json|QUALITY.json.*checksum"
    ):
        verify_release(release_dir)


def test_current_pointer_cannot_escape_release_root(tmp_path: Path) -> None:
    releases_dir = tmp_path / "releases"
    _write_release(tmp_path / "outside")
    (releases_dir / "CURRENT").parent.mkdir(parents=True)
    (releases_dir / "CURRENT").write_text("../outside\n", encoding="utf-8")

    with pytest.raises(ReleaseIntegrityError, match="pointer|release id|outside"):
        resolve_release_path(releases_dir / "CURRENT")


def test_verify_release_rejects_chunk_id_mismatch(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "release-1")
    ids_path = release_dir / "chunk_ids.json"
    ids_path.write_text(json.dumps(["not-the-catalog-chunk"]), encoding="utf-8")
    manifest = _read_manifest(release_dir)
    ids = ids_path.read_bytes()
    manifest["files"]["chunk_ids.json"] = {
        "sha256": hashlib.sha256(ids).hexdigest(),
        "bytes": len(ids),
    }
    _write_manifest(release_dir, manifest)

    with pytest.raises(ReleaseIntegrityError, match="chunk.*(catalog|id)|id.*chunk"):
        verify_release(release_dir)


def test_manifest_must_declare_cosine_distance_for_normalized_vectors(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "release-a")
    manifest = _read_manifest(release_dir)
    manifest["embedding"]["distance_metric"] = "l2"
    _write_manifest(release_dir, manifest)

    with pytest.raises(ReleaseIntegrityError, match="distance_metric must be cosine"):
        verify_release(release_dir)


@pytest.mark.parametrize(
    ("dimension", "vector_count", "message"),
    [(3, 1, "Vector dimension mismatch"), (1024, 0, "FAISS vector count")],
)
def test_release_rejects_faiss_dimension_or_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    vector_count: int,
    message: str,
) -> None:
    release_dir = _write_release(tmp_path / "release-a")

    class IndexFlatIP:
        d = dimension
        ntotal = vector_count

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(read_index=lambda _path: IndexFlatIP()),
    )

    with pytest.raises(ReleaseIntegrityError, match=message):
        verify_release(release_dir)


def test_release_rejects_wrong_faiss_index_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_dir = _write_release(tmp_path / "release-a")

    class IndexFlatL2:
        d = 1024
        ntotal = 1

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(read_index=lambda _path: IndexFlatL2()),
    )

    with pytest.raises(ReleaseIntegrityError, match="must be IndexFlatIP"):
        verify_release(release_dir)


def test_container_mounts_configured_current_pointer(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")
    pointer = release_dir.parent / "CURRENT"
    pointer.write_text(release_dir.name + "\n", encoding="utf-8")
    settings = Settings.from_env({}, project_root=tmp_path, release_path=pointer)

    container = ApplicationContainer(settings)

    try:
        assert container.is_mounted
        assert container.release_dir == release_dir.resolve()
        with pytest.raises(sqlite3.OperationalError):
            container.catalog.conn.execute(
                "INSERT INTO logical_documents (id) VALUES ('write-attempt')"
            )
    finally:
        container.close()


def test_container_constructs_hybrid_adapters_lazily_without_loading_model_weights(
    tmp_path: Path,
) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")
    container = ApplicationContainer(release_dir=release_dir)

    assert container._hybrid_retrieval is None
    service = container.hybrid_retrieval

    assert service.__class__.__name__ == "HybridRetrieval"
    assert service.vector_index.encoder._model is None
    assert container._hybrid_retrieval is service
    assert container._serving_decision_engine is None
    try:
        decision_engine = container.serving_decision_engine
        assert decision_engine.__class__.__name__ == "DecisionEngine"
        assert container._serving_decision_engine is decision_engine
    finally:
        container.close()
    assert container._hybrid_retrieval is None
    assert container._serving_decision_engine is None


def test_mounted_release_warmup_uses_only_read_only_serving_adapters(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")
    container = ApplicationContainer(
        release_dir=release_dir,
        explanation_override=object(),
    )

    try:
        state = container.warmup()

        assert state["ready"] is True
        assert state["model_loaded"] is False
        assert container._retrieval_service is None
        assert container._decision_engine is None
        assert container._hybrid_retrieval is not None
        assert container._serving_decision_engine is not None
        assert container.hybrid_retrieval.vector_index.encoder._model is None
    finally:
        container.close()


def test_configured_bge_reranker_is_lazy_until_first_analysis(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")
    settings = Settings.from_env(
        {"SPEC_CIPHER_RERANKER_ENABLED": "true"},
        project_root=tmp_path,
        release_path=release_dir,
    )
    container = ApplicationContainer(settings=settings, release_dir=release_dir)

    try:
        engine = container.serving_decision_engine
        assert engine.reranker.__class__.__name__ == "BGERerankerAdapter"
        assert engine.reranker._adapter is None
        assert engine.reranker._options["local_files_only"] is True
    finally:
        container.close()


def test_container_fails_startup_for_corrupt_configured_release(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "release-1")
    (release_dir / "QUALITY.json").write_text("tampered", encoding="utf-8")
    settings = Settings.from_env({}, project_root=tmp_path, release_path=release_dir)

    with pytest.raises(ReleaseIntegrityError, match="SHA-256 mismatch"):
        ApplicationContainer(settings)


def test_publisher_promotes_verified_release_and_updates_current(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    releases_root = tmp_path / "releases"
    staging_release = _write_release(staging_root / "release-1")
    publisher = ReleasePublisher(staging_root, releases_root)

    published = publisher.publish(staging_release)

    assert published == releases_root / "release-1"
    assert published.is_dir()
    assert not staging_release.exists()
    pointer = json.loads((releases_root / "CURRENT").read_text(encoding="utf-8"))
    assert pointer["release_id"] == "release-1"
    assert pointer["release_sha256"].startswith("sha256:")
    assert resolve_release_path(releases_root / "CURRENT") == published.resolve()


def test_publisher_does_not_promote_corrupt_staging_release(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    releases_root = tmp_path / "releases"
    staging_release = _write_release(staging_root / "release-1")
    (staging_release / "QUALITY.json").write_text("tampered", encoding="utf-8")
    publisher = ReleasePublisher(staging_root, releases_root)

    with pytest.raises(ReleasePublishError, match="failed verification"):
        publisher.publish(staging_release)

    assert staging_release.is_dir()
    assert not (releases_root / "release-1").exists()
    assert not (releases_root / "CURRENT").exists()


def test_publisher_rolls_back_to_verified_release(tmp_path: Path) -> None:
    releases_root = tmp_path / "releases"
    _write_release(releases_root / "release-1")
    _write_release(releases_root / "release-2")
    publisher = ReleasePublisher(tmp_path / "staging", releases_root)
    (releases_root / "CURRENT").write_text("release-2\n", encoding="utf-8")

    selected = publisher.rollback("release-1")

    assert selected == releases_root / "release-1"
    pointer = json.loads((releases_root / "CURRENT").read_text(encoding="utf-8"))
    assert pointer["release_id"] == "release-1"


def test_current_pointer_authenticates_release_manifest(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path / "releases" / "release-1")
    publisher = ReleasePublisher(tmp_path / "staging", release_dir.parent)
    publisher.rollback("release-1")
    current_path = release_dir.parent / "CURRENT"

    pointer = json.loads(current_path.read_text(encoding="utf-8"))
    assert (
        pointer["release_sha256"]
        == "sha256:" + hashlib.sha256((release_dir / "release.json").read_bytes()).hexdigest()
    )
    (release_dir / "release.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReleaseIntegrityError, match="does not match release.json"):
        resolve_release_path(current_path)


@pytest.mark.parametrize("release_id", [None, 1, "..", "../escape", "C:drive-relative"])
def test_publisher_rejects_invalid_rollback_ids(tmp_path: Path, release_id: Any) -> None:
    publisher = ReleasePublisher(tmp_path / "staging", tmp_path / "releases")

    with pytest.raises(ReleasePublishError, match="Invalid release ID"):
        publisher.rollback(release_id)


def test_publisher_refuses_to_replace_an_existing_immutable_release(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    releases_root = tmp_path / "releases"
    _write_release(releases_root / "release-1")
    staging_release = _write_release(staging_root / "release-1")
    publisher = ReleasePublisher(staging_root, releases_root)

    with pytest.raises(ReleasePublishError, match="immutable"):
        publisher.publish(staging_release)

    assert staging_release.is_dir()
