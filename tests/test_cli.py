from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from speccipher.cli import manage
from speccipher.config.settings import Settings
from speccipher.release.manifest import ReleaseIntegrityError


def test_verify_defaults_to_configured_release_path(monkeypatch, tmp_path: Path, capsys) -> None:
    current = tmp_path / "CURRENT"
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(
                paths=SimpleNamespace(project_root=tmp_path, release_path=current)
            )
        ),
    )
    seen: list[Path] = []
    monkeypatch.setattr(
        manage, "verify_release", lambda path: seen.append(Path(path)) or {"release_id": "r-1"}
    )

    result = manage.main(["verify"])

    assert result == 0
    assert seen == [current]
    assert '"release_id": "r-1"' in capsys.readouterr().out


def test_verify_reports_integrity_failure_without_traceback(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(
                paths=SimpleNamespace(project_root=tmp_path, release_path=tmp_path / "CURRENT")
            )
        ),
    )

    def fail_verification(_path: Path) -> None:
        raise ReleaseIntegrityError("payload digest mismatch")

    monkeypatch.setattr(manage, "verify_release", fail_verification)

    result = manage.main(["verify"])

    output = capsys.readouterr()
    assert result == 2
    assert "payload digest mismatch" in output.err
    assert output.out == ""


def test_manifest_command_targets_staging_release_and_prints_manifest(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(
                paths=SimpleNamespace(project_root=tmp_path, release_path=tmp_path / "CURRENT")
            )
        ),
    )
    seen: list[tuple[Path, dict[str, object]]] = []

    def record_manifest(path: Path, **options: object) -> dict[str, object]:
        seen.append((Path(path), options))
        return {"release_id": Path(path).name}

    monkeypatch.setattr(manage, "write_release_manifest", record_manifest)
    result = manage.main(
        [
            "manifest",
            "release-1",
            "--embedding-model",
            "model-id",
            "--embedding-revision",
            "revision-id",
            "--chunking-version",
            "2.0",
            "--git-sha",
            "abc1234",
        ]
    )

    assert result == 0
    assert seen == [
        (
            tmp_path / "data" / "staging" / "release-1",
            {
                "embedding_model": "model-id",
                "embedding_revision": "revision-id",
                "git_sha": "abc1234",
                "chunking_version": "2.0",
            },
        )
    ]
    assert '"release_id": "release-1"' in capsys.readouterr().out


def test_index_command_uses_configured_encoder_and_staged_jsonl(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    stage = tmp_path / "data" / "staging" / "release-1"
    stage.mkdir(parents=True)
    (stage / "chunks.jsonl").write_text('{"chunk_id":"c1","text":"some content"}\n')
    retrieval = SimpleNamespace(
        embedding_dimension=1024,
        normalize_embeddings=True,
        embedding_model="model-id",
        batch_size=4,
        embedding_device="cpu",
        allow_cpu_fallback=True,
        max_seq_length=128,
        embedding_use_fp16=False,
        embedding_model_version="revision-id",
        hf_offline=True,
        allow_model_download=False,
    )
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(
                paths=SimpleNamespace(project_root=tmp_path),
                retrieval=retrieval,
            )
        ),
    )
    from retrieval.embeddings import model

    instances = []

    class FakeEncoder:
        def __init__(self, **options: object) -> None:
            self.options = options
            self.released = False
            instances.append(self)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(model, "SentenceTransformerEncoder", FakeEncoder)
    seen = []

    def build(records, encoder, vectors_path, ids_path, **options):
        seen.append((list(records), encoder, Path(vectors_path), Path(ids_path), options))
        return 1

    monkeypatch.setattr(manage, "build_dense_payloads", build)

    result = manage.main(["index", "release-1"])

    assert result == 0
    assert len(seen) == 1
    assert seen[0][0] == [{"chunk_id": "c1", "text": "some content"}]
    assert seen[0][2] == stage / "vectors.faiss"
    assert seen[0][3] == stage / "chunk_ids.json"
    assert seen[0][4] == {"dimension": 1024, "batch_size": 4}
    assert instances[0].released is True
    assert '"vectors": 1' in capsys.readouterr().out


def test_stage_commands_reject_path_traversal(tmp_path: Path) -> None:
    from speccipher.cli.manage import _staging_path

    with pytest.raises(ValueError, match="Invalid staged release ID"):
        _staging_path(tmp_path / "data", "../outside")


def test_catalog_command_builds_both_release_payloads(monkeypatch, tmp_path: Path, capsys) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text('{"documents":[],"source_files":[],"chunks":[]}', encoding="utf-8")
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(
            lambda cls: SimpleNamespace(
                paths=SimpleNamespace(project_root=tmp_path, release_path=tmp_path / "CURRENT")
            )
        ),
    )
    seen = []

    def build(catalog_path, chunks_path, **records):
        seen.append((Path(catalog_path), Path(chunks_path), records))
        return Path(catalog_path), Path(chunks_path)

    monkeypatch.setattr(manage, "build_catalog_payloads", build)

    result = manage.main(["catalog", "release-1", str(records_path)])

    stage = tmp_path / "data" / "staging" / "release-1"
    assert result == 0
    assert seen == [
        (
            stage / "catalog.sqlite",
            stage / "chunks.jsonl",
            {
                "documents": [],
                "source_files": [],
                "chunks": [],
                "relationships": [],
                "scope_texts": [],
                "operator_overrides": [],
                "missing_volumes": [],
            },
        )
    ]
    output = capsys.readouterr().out
    assert '"catalog":' in output
    assert '"chunks":' in output
