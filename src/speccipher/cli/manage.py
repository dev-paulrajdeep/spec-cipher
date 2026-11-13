"""Operational commands for verifying and selecting immutable releases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from speccipher import __version__
from speccipher.config.settings import Settings
from speccipher.offline.indexing.faiss_build import build_dense_payloads
from speccipher.offline.indexing.fts_build import build_catalog_payloads
from speccipher.offline.publishing.manifest import write_release_manifest
from speccipher.release.manifest import ReleaseIntegrityError, verify_release
from speccipher.release.publisher import ReleasePublisher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speccipher")
    parser.add_argument("--version", action="version", version=f"SpecCipher {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify", help="verify a release or the configured CURRENT release"
    )
    verify.add_argument("release", nargs="?", help="release directory or CURRENT pointer")

    publish = commands.add_parser("publish", help="verify and atomically publish a staged release")
    publish.add_argument("release_id", help="staged release directory name")

    index = commands.add_parser(
        "index", help="encode staged chunks and build vectors.faiss plus chunk_ids.json"
    )
    index.add_argument("release_id", help="staged release directory name")

    catalog = commands.add_parser(
        "catalog", help="build catalog.sqlite from prepared catalog records"
    )
    catalog.add_argument("release_id", help="staged release directory name")
    catalog.add_argument(
        "records", type=Path, help="JSON file with documents, source_files, and chunks"
    )

    manifest = commands.add_parser(
        "manifest", help="write a checksummed manifest for a complete staged release"
    )
    manifest.add_argument("release_id", help="staged release directory name")
    manifest.add_argument("--embedding-model", required=True)
    manifest.add_argument("--embedding-revision", required=True)
    manifest.add_argument("--chunking-version", required=True)
    manifest.add_argument("--git-sha")

    rollback = commands.add_parser("rollback", help="point CURRENT to a verified release")
    rollback.add_argument("release_id", help="previously published release directory name")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _staging_path(data_root: Path, release_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id) is None:
        raise ValueError(f"Invalid staged release ID: {release_id!r}")
    staging_root = (data_root / "staging").resolve()
    target = (staging_root / release_id).resolve()
    if target.parent != staging_root:
        raise ValueError("Staged release path must be a direct child of data/staging")
    return target


def _read_chunk_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path.name} line {line_number}")
            yield value


def _read_catalog_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read catalog records from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Catalog records file must contain a JSON object")
    record_names = (
        "documents", "source_files", "chunks", "relationships", "scope_texts",
        "operator_overrides", "missing_volumes",
    )
    records: dict[str, list[dict[str, Any]]] = {}
    for name in record_names:
        rows = payload.get(name, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Catalog records field {name!r} must be an array of objects")
        records[name] = rows
    for required in ("documents", "source_files", "chunks"):
        if required not in payload:
            raise ValueError(f"Catalog records file is missing required field {required!r}")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Run one release-management command and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    data_root = settings.paths.project_root / "data"
    publisher = ReleasePublisher(data_root / "staging", data_root / "releases")

    try:
        if args.command == "verify":
            release_path = Path(args.release) if args.release else settings.paths.release_path
            _print_json(verify_release(release_path))
        elif args.command == "publish":
            published = publisher.publish(_staging_path(data_root, args.release_id))
            _print_json(verify_release(published))
        elif args.command == "index":
            stage_path = _staging_path(data_root, args.release_id)
            retrieval = settings.retrieval
            dimension = retrieval.embedding_dimension or 1024
            if dimension != 1024:
                raise ValueError(
                    f"Corpus releases require 1024-dimensional embeddings, got {dimension}"
                )
            if not retrieval.normalize_embeddings:
                raise ValueError("Corpus release embeddings must be unit-normalized")
            from retrieval.embeddings.model import SentenceTransformerEncoder

            encoder = SentenceTransformerEncoder(
                model_name=retrieval.embedding_model,
                batch_size=retrieval.batch_size,
                normalize_embeddings=True,
                device=retrieval.embedding_device,
                allow_cpu_fallback=retrieval.allow_cpu_fallback,
                max_seq_length=retrieval.max_seq_length,
                embedding_dimension=dimension,
                use_fp16=retrieval.embedding_use_fp16,
                model_version=retrieval.embedding_model_version,
                local_files_only=(retrieval.hf_offline or not retrieval.allow_model_download),
            )
            try:
                count = build_dense_payloads(
                    _read_chunk_records(stage_path / "chunks.jsonl"),
                    encoder,
                    stage_path / "vectors.faiss",
                    stage_path / "chunk_ids.json",
                    dimension=dimension,
                    batch_size=retrieval.batch_size,
                )
            finally:
                encoder.release()
            _print_json({"release_id": args.release_id, "vectors": count})
        elif args.command == "catalog":
            stage_path = _staging_path(data_root, args.release_id)
            records = _read_catalog_records(args.records)
            catalog_path, chunks_path = build_catalog_payloads(
                stage_path / "catalog.sqlite",
                stage_path / "chunks.jsonl",
                **records,
            )
            _print_json(
                {
                    "release_id": args.release_id,
                    "catalog": str(catalog_path),
                    "chunks": str(chunks_path),
                }
            )
        elif args.command == "manifest":
            manifest = write_release_manifest(
                _staging_path(data_root, args.release_id),
                embedding_model=args.embedding_model,
                embedding_revision=args.embedding_revision,
                git_sha=args.git_sha,
                chunking_version=args.chunking_version,
            )
            _print_json(manifest)
        elif args.command == "rollback":
            rolled_back = publisher.rollback(args.release_id)
            _print_json(verify_release(rolled_back))
        else:
            parser.error(f"unsupported command: {args.command}")
    except (OSError, ValueError, RuntimeError, ReleaseIntegrityError) as exc:
        print(f"speccipher {args.command}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
