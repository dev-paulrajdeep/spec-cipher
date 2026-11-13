"""SQLite Catalog Repository (Layer 5 Catalog)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


class CatalogRepository:
    """Read-only SQLite catalog repository for serving plane metadata and lexical search."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Catalog SQLite database not found: {self.db_path}")
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only = ON")

    def close(self) -> None:
        """Close SQLite database connection to release file locks."""
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass

    def search_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Perform BM25 lexical search using SQLite FTS5 table."""
        if limit is None or limit <= 0:
            safe_limit = 10
        else:
            safe_limit = min(int(limit), 100)

        STANDARD_PREFIXES = {"IS", "SP", "IRC", "NBC", "BS", "EN", "ISO", "ASTM"}
        ENGLISH_STOPWORDS = {
            "for",
            "and",
            "the",
            "in",
            "of",
            "to",
            "a",
            "with",
            "as",
            "on",
            "by",
            "an",
            "at",
            "from",
        }
        cleaned = re.sub(r"[^\w\s]", " ", query)
        terms: list[str] = []
        for t in cleaned.split():
            t_upper = t.upper()
            t_lower = t.lower()
            if t_upper in STANDARD_PREFIXES:
                terms.append(t_upper)
            elif t.isdigit():
                terms.append(t)
            elif len(t) >= 3 and t_lower not in ENGLISH_STOPWORDS:
                terms.append(t)

        if not terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms[:8])
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.chunk_id, c.logical_doc_id, c.standard_code, c.source_file_id, c.absolute_page_start,
                   c.absolute_page_end, c.section_number, c.section_title, c.text, c.is_eligible,
                   bm25(chunks_fts) as rank_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
            """,
            (fts_query, safe_limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch provenance records for known chunk IDs in caller order-independent form."""
        if not chunk_ids:
            return {}
        records: dict[str, dict[str, Any]] = {}
        cursor = self.conn.cursor()
        for offset in range(0, len(chunk_ids), 500):
            batch = chunk_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(
                f"""
                SELECT c.chunk_id, c.logical_doc_id, c.standard_code, c.source_file_id,
                       COALESCE(s.file_name, s.relative_path, s.id) AS source_file,
                       c.absolute_page_start, c.absolute_page_end, c.section_number,
                       c.section_title, c.text, c.is_eligible
                FROM chunks AS c
                JOIN source_files AS s ON s.id = c.source_file_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                batch,
            )
            records.update((row["chunk_id"], dict(row)) for row in cursor.fetchall())
        return records

    def get_standards(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve logical standard documents matching optional query prefix."""
        cur = self.conn.cursor()
        if query:
            cur.execute(
                """
                SELECT id, family, number, part, year, code, title, status, edition, is_eligible, is_stub, page_count
                FROM logical_documents
                WHERE code LIKE ? OR title LIKE ?
                ORDER BY code ASC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            )
        else:
            cur.execute(
                """
                SELECT id, family, number, part, year, code, title, status, edition, is_eligible, is_stub, page_count
                FROM logical_documents
                ORDER BY code ASC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]

    def get_standard_by_code(self, code: str) -> dict[str, Any] | None:
        """Lookup a standard by exact code or normalized code."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, family, number, part, year, code, title, status, edition, is_eligible, is_stub, page_count
            FROM logical_documents
            WHERE code = ? OR id = ?
            LIMIT 1
            """,
            (code, code),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_relationships(self, source_code: str) -> list[dict[str, Any]]:
        """Retrieve directed relationships originating from source_code."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT source_code, target_code, relationship_type, page_number, evidence_text
            FROM relationships
            WHERE source_code = ?
            ORDER BY target_code ASC
            """,
            (source_code,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_scope_text(self, standard_code: str) -> dict[str, Any] | None:
        """Retrieve pre-compiled scope text and exclusions."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT standard_code, scope, exclusions
            FROM scope_text
            WHERE standard_code = ?
            LIMIT 1
            """,
            (standard_code,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
