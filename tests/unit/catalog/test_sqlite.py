from __future__ import annotations

import sqlite3

from speccipher.catalog.sqlite import CatalogRepository


def test_chunk_lookup_returns_catalog_source_filename(tmp_path) -> None:
    database = tmp_path / "catalog.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE source_files (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                relative_path TEXT
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                logical_doc_id TEXT,
                standard_code TEXT,
                source_file_id TEXT,
                absolute_page_start INTEGER,
                absolute_page_end INTEGER,
                section_number TEXT,
                section_title TEXT,
                text TEXT,
                is_eligible INTEGER
            );
            INSERT INTO source_files VALUES ('source-1', 'official.pdf', 'sources/official.pdf');
            INSERT INTO chunks VALUES (
                'chunk-1', 'doc-1', 'IS 456:2000', 'source-1',
                12, 13, '4.1', 'Scope', 'Verified source text.', 1
            );
            """
        )

    repository = CatalogRepository(database)
    try:
        record = repository.get_chunks_by_ids(["chunk-1"])["chunk-1"]
        assert record["source_file"] == "official.pdf"
        assert record["text"] == "Verified source text."
    finally:
        repository.close()
