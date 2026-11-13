"""SQLite FTS5 lexical adapter preserving raw BM25 scores."""

from __future__ import annotations

import math
from typing import Any, Protocol

from speccipher.domain.candidates import Candidate
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk


class CatalogSearch(Protocol):
    """Read-only lexical search operation exposed by the catalog repository."""

    def search_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]: ...


class FTS5LexicalIndex:
    """Adapt CatalogRepository FTS rows to typed domain candidates."""

    def __init__(self, catalog: CatalogSearch) -> None:
        self.catalog = catalog

    @staticmethod
    def _candidate(record: dict[str, Any]) -> Candidate:
        raw_score = record.get("rank_score")
        if raw_score is None:
            raise ValueError("FTS result is missing the raw BM25 rank_score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("FTS result has a non-finite BM25 rank_score")

        code = record.get("standard_code")
        standard_id = None
        if isinstance(code, str) and code.strip():
            try:
                standard_id = StandardId.parse(code)
            except ValueError:
                standard_id = None

        chunk = DocumentChunk(
            chunk_id=str(record["chunk_id"]),
            standard_id=standard_id,
            logical_doc_id=str(record["logical_doc_id"]),
            source_file_id=str(record["source_file_id"]),
            absolute_page_start=int(record["absolute_page_start"]),
            absolute_page_end=int(record["absolute_page_end"]),
            section_number=record.get("section_number"),
            section_title=record.get("section_title"),
            text=str(record.get("text") or ""),
            is_eligible=bool(record.get("is_eligible", False)),
        )
        return Candidate(chunk=chunk, lexical_score=score)

    def search(self, query: str, limit: int) -> tuple[Candidate, ...]:
        """Return BM25-ranked candidates, retaining the unnormalized BM25 score."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        rows = self.catalog.search_fts(query, limit=limit)
        return tuple(self._candidate(row) for row in rows)
