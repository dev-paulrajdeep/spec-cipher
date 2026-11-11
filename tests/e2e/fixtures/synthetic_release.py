"""Synthetic Corpus Release Builder for E2E tests.

Constructs valid and deliberately corrupted Schema 2.0 Corpus Releases:
- data/releases/<release_id>/
  - release.json (Schema 2.0)
  - catalog.sqlite (FTS5 table, relationships, scope, chunks)
  - chunks.jsonl
  - vectors.faiss
  - chunk_ids.json
  - QUALITY.json
- data/releases/CURRENT pointer
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from tests.e2e.fixtures.sample_data import (
    SAMPLE_CHUNKS,
    SAMPLE_RELATIONSHIPS,
    SAMPLE_SCOPE_TEXTS,
    SAMPLE_STANDARDS,
)

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SyntheticCorpusReleaseBuilder:
    """Builds valid or deliberately corrupted Schema 2.0 Corpus Releases for testing."""

    def __init__(self, root_dir: Path, release_id: str = "20261111T093000Z-synth01"):
        self.root_dir = Path(root_dir)
        self.release_id = release_id
        self.releases_base = self.root_dir / "data" / "releases"
        self.release_dir = self.releases_base / release_id
        self.chunks_data = list(SAMPLE_CHUNKS)
        self.standards_data = list(SAMPLE_STANDARDS)
        self.relationships_data = list(SAMPLE_RELATIONSHIPS)
        self.dim = 1024

    def build_valid_release(self) -> Path:
        """Build a fully valid Schema 2.0 corpus release."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self._build_catalog_sqlite()
        self._build_chunks_jsonl()
        self._build_chunk_ids_json()
        self._build_faiss_index()
        self._build_quality_report()
        self._build_manifest(corrupt_file=None)
        self.set_current_pointer(self.release_id)
        return self.release_dir

    def build_corrupt_checksum_release(self) -> Path:
        """Build a release with tampered checksum for catalog.sqlite."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self._build_catalog_sqlite()
        self._build_chunks_jsonl()
        self._build_chunk_ids_json()
        self._build_faiss_index()
        self._build_quality_report()
        self._build_manifest(corrupt_file="catalog.sqlite")
        self.set_current_pointer(self.release_id)
        return self.release_dir

    def build_mismatched_counts_release(self) -> Path:
        """Build a release where manifest chunk count != vector count."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self._build_catalog_sqlite()
        self._build_chunks_jsonl()
        self._build_chunk_ids_json()
        self._build_faiss_index()
        self._build_quality_report()
        self._build_manifest(corrupt_counts=True)
        self.set_current_pointer(self.release_id)
        return self.release_dir

    def build_mismatched_dim_release(self) -> Path:
        """Build a release where manifest dim does not match faiss index."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self._build_catalog_sqlite()
        self._build_chunks_jsonl()
        self._build_chunk_ids_json()
        self._build_faiss_index(custom_dim=384)
        self._build_quality_report()
        self._build_manifest(manifest_dim=1024)
        self.set_current_pointer(self.release_id)
        return self.release_dir

    def build_missing_file_release(self, missing_filename: str) -> Path:
        """Build a release missing a required constituent file."""
        self.build_valid_release()
        target = self.release_dir / missing_filename
        if target.exists():
            target.unlink()
        return self.release_dir

    def set_current_pointer(self, target_id: str) -> None:
        """Atomically set data/releases/CURRENT pointer."""
        self.releases_base.mkdir(parents=True, exist_ok=True)
        current_file = self.releases_base / "CURRENT"
        temp_file = self.releases_base / f".CURRENT.tmp.{os.getpid()}"
        temp_file.write_text(f"{target_id}\n", encoding="utf-8")
        os.replace(temp_file, current_file)

    def rollback_to(self, target_id: str) -> None:
        """Instant rollback by pointing CURRENT to target_id."""
        target_dir = self.releases_base / target_id
        if not target_dir.exists():
            raise FileNotFoundError(f"Rollback target release {target_id} does not exist at {target_dir}")
        self.set_current_pointer(target_id)

    def _build_catalog_sqlite(self) -> Path:
        db_path = self.release_dir / "catalog.sqlite"
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.executescript("""
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
                is_eligible INTEGER DEFAULT 1,
                is_stub INTEGER DEFAULT 0,
                page_count INTEGER DEFAULT 0
            );

            CREATE TABLE source_files (
                id TEXT PRIMARY KEY,
                logical_doc_id TEXT REFERENCES logical_documents(id),
                file_name TEXT,
                relative_path TEXT,
                sha256 TEXT,
                page_count INTEGER,
                part_index INTEGER,
                page_offset INTEGER DEFAULT 0,
                extraction_status TEXT DEFAULT 'processed'
            );

            CREATE TABLE sections (
                id TEXT PRIMARY KEY,
                logical_doc_id TEXT REFERENCES logical_documents(id),
                section_number TEXT,
                section_title TEXT,
                level INTEGER DEFAULT 1,
                page_start INTEGER,
                page_end INTEGER,
                text TEXT
            );

            CREATE TABLE relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_code TEXT,
                target_code TEXT,
                relationship_type TEXT,
                page_number INTEGER,
                evidence_text TEXT
            );

            CREATE TABLE scope_text (
                standard_code TEXT PRIMARY KEY,
                scope TEXT,
                exclusions TEXT
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                logical_doc_id TEXT REFERENCES logical_documents(id),
                standard_code TEXT,
                source_file_id TEXT REFERENCES source_files(id),
                page_start INTEGER,
                page_end INTEGER,
                absolute_page_start INTEGER,
                absolute_page_end INTEGER,
                section_number TEXT,
                section_title TEXT,
                text TEXT,
                is_eligible INTEGER DEFAULT 1
            );

            CREATE VIRTUAL TABLE chunks_fts USING fts5 (
                chunk_id UNINDEXED,
                standard_code,
                section_number,
                section_title,
                text,
                tokenize='unicode61'
            );
        """)

        # Populate logical_documents and source_files
        for std in self.standards_data:
            cur.execute(
                """
                INSERT INTO logical_documents 
                (id, family, number, part, year, code, title, status, edition, identity_source, is_eligible, is_stub, page_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    std["id"],
                    std["family"],
                    std["number"],
                    std.get("part"),
                    std.get("year"),
                    std["code"],
                    std["title"],
                    std["status"],
                    std.get("edition"),
                    std.get("identity_source", "extracted"),
                    std.get("is_eligible", 1),
                    std.get("is_stub", 0),
                    std.get("page_count", 0),
                ),
            )
            for vol in std.get("volumes", []):
                file_id = vol["file_name"]
                cur.execute(
                    """
                    INSERT INTO source_files
                    (id, logical_doc_id, file_name, relative_path, sha256, page_count, part_index, page_offset, extraction_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        std["id"],
                        vol["file_name"],
                        vol["file_name"],
                        hashlib.sha256(vol["file_name"].encode()).hexdigest(),
                        vol["page_count"],
                        vol["part_index"],
                        vol["page_offset"],
                        "processed",
                    ),
                )

        # Populate relationships
        for rel in self.relationships_data:
            cur.execute(
                """
                INSERT INTO relationships (source_code, target_code, relationship_type, page_number, evidence_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rel["source_code"],
                    rel["target_code"],
                    rel["relationship_type"],
                    rel.get("page_number", 1),
                    rel.get("evidence_text", ""),
                ),
            )

        # Populate scope_text
        for code, scope in SAMPLE_SCOPE_TEXTS.items():
            cur.execute("INSERT INTO scope_text (standard_code, scope, exclusions) VALUES (?, ?, ?)", (code, scope, ""))

        # Populate chunks and chunks_fts
        for chk in self.chunks_data:
            cur.execute(
                """
                INSERT INTO chunks
                (chunk_id, logical_doc_id, standard_code, source_file_id, page_start, page_end,
                 absolute_page_start, absolute_page_end, section_number, section_title, text, is_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chk["chunk_id"],
                    chk["logical_doc_id"],
                    chk["standard_code"],
                    chk["source_file_id"],
                    chk["page_start"],
                    chk["page_end"],
                    chk["absolute_page_start"],
                    chk["absolute_page_end"],
                    chk.get("section_number"),
                    chk.get("section_title"),
                    chk["text"],
                    chk.get("is_eligible", 1),
                ),
            )
            cur.execute(
                """
                INSERT INTO chunks_fts (chunk_id, standard_code, section_number, section_title, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chk["chunk_id"],
                    chk["standard_code"],
                    chk.get("section_number", ""),
                    chk.get("section_title", ""),
                    chk["text"],
                ),
            )

        conn.commit()
        conn.close()
        return db_path

    def _build_chunks_jsonl(self) -> Path:
        path = self.release_dir / "chunks.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for chk in self.chunks_data:
                f.write(json.dumps(chk, ensure_ascii=False) + "\n")
        return path

    def _build_chunk_ids_json(self) -> Path:
        path = self.release_dir / "chunk_ids.json"
        chunk_ids = [chk["chunk_id"] for chk in self.chunks_data]
        path.write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")
        return path

    def _build_faiss_index(self, custom_dim: int | None = None) -> Path:
        path = self.release_dir / "vectors.faiss"
        dim = custom_dim or self.dim
        n = len(self.chunks_data)

        if HAS_FAISS:
            index = faiss.IndexFlatIP(dim)
            # Create deterministic unit-norm vectors for testing
            rng = np.random.RandomState(42)
            vectors = rng.randn(n, dim).astype("float32")
            # Normalize to unit length
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            index.add(vectors)
            faiss.write_index(index, str(path))
        else:
            # Fallback binary placeholder if faiss not loaded
            placeholder = b"FAISS_INDEX_SYNTH_" + (b"\x00" * (dim * n * 4))
            path.write_bytes(placeholder)

        return path

    def _build_quality_report(self) -> Path:
        path = self.release_dir / "QUALITY.json"
        report = {
            "release_id": self.release_id,
            "total_documents": len(self.standards_data),
            "total_chunks": len(self.chunks_data),
            "eligible_chunks": sum(1 for c in self.chunks_data if c.get("is_eligible", 1)),
            "unresolved_documents": 0,
            "stubs_count": sum(1 for s in self.standards_data if s.get("is_stub", 0)),
        }
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def _build_manifest(
        self,
        corrupt_file: str | None = None,
        corrupt_counts: bool = False,
        manifest_dim: int | None = None,
    ) -> Path:
        path = self.release_dir / "release.json"
        files_map = {}
        for fname in ["catalog.sqlite", "chunks.jsonl", "vectors.faiss", "chunk_ids.json"]:
            fpath = self.release_dir / fname
            if fpath.exists():
                digest = compute_file_sha256(fpath)
                size = fpath.stat().st_size
                if corrupt_file == fname:
                    # Invert last characters of hex to corrupt checksum
                    digest = digest[:-4] + "0000"
                files_map[fname] = {"sha256": digest, "bytes": size}

        chunk_count = len(self.chunks_data)
        vector_count = chunk_count if not corrupt_counts else chunk_count + 10

        manifest = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "schema_version": "2.0",
            "release_id": self.release_id,
            "created_at": "2026-11-11T09:30:00Z",
            "git_sha": "b5f6e79a8b1c",
            "chunking_version": "1.1",
            "embedding": {
                "model": "BAAI/bge-m3",
                "revision": "main",
                "dim": manifest_dim or self.dim,
                "normalized": True,
            },
            "index_type": "IndexFlatIP",
            "counts": {
                "documents": len(self.standards_data),
                "eligible_documents": sum(1 for s in self.standards_data if s.get("is_eligible", 1)),
                "chunks": chunk_count,
                "eligible_chunks": sum(1 for c in self.chunks_data if c.get("is_eligible", 1)),
                "vectors": vector_count,
            },
            "sources": [
                {
                    "path": "SP 7_2026-1-400.pdf",
                    "sha256": "4a7b9c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
                    "pages": 400,
                    "disposition": "processed",
                },
                {
                    "path": "SP 7_2026-401-800.pdf",
                    "sha256": "4a7b9c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
                    "pages": 400,
                    "disposition": "override",
                },
            ],
            "files": files_map,
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path
