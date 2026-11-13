"""Lazy FAISS adapter for immutable release vector indexes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from speccipher.domain.candidates import Candidate
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk


class Encoder(Protocol):
    """Query encoder supplied by the composition root."""

    @property
    def dimension(self) -> int: ...

    def encode_text(self, text: str) -> Any: ...


class ChunkCatalog(Protocol):
    """Read-only chunk provenance lookup supplied by the catalog adapter."""

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]: ...


class FaissVectorIndex:
    """Exact inner-product search against a release's FAISS index and chunk map.

    FAISS and NumPy stay behind the search lifecycle so importing this module
    does not load native libraries. Model loading remains owned by ``Encoder``.
    """

    def __init__(
        self,
        index_path: Path | str,
        chunk_ids_path: Path | str,
        encoder: Encoder,
        catalog: ChunkCatalog,
        *,
        expected_dimension: int = 1024,
    ) -> None:
        if expected_dimension < 1:
            raise ValueError("expected_dimension must be positive")
        self.index_path = Path(index_path)
        self.chunk_ids_path = Path(chunk_ids_path)
        self.encoder = encoder
        self.catalog = catalog
        self.expected_dimension = expected_dimension
        self._lock = RLock()
        self._index: Any = None
        self._chunk_ids: tuple[str, ...] | None = None

    def _ensure_loaded(self) -> None:
        if self._index is not None and self._chunk_ids is not None:
            return
        with self._lock:
            if self._index is not None and self._chunk_ids is not None:
                return
            if not self.index_path.is_file() or not self.chunk_ids_path.is_file():
                raise FileNotFoundError("Release vector index or chunk_ids.json is missing")

            try:
                chunk_ids = json.loads(self.chunk_ids_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid release chunk_ids.json: {exc}") from exc
            if (
                not isinstance(chunk_ids, list)
                or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids)
            ):
                raise ValueError("chunk_ids.json must contain unique non-empty string IDs")

            try:
                import faiss

                index = faiss.read_index(str(self.index_path))
            except Exception as exc:
                raise RuntimeError(f"Could not load release FAISS index: {exc}") from exc
            if index.d != self.expected_dimension:
                raise ValueError(
                    f"FAISS dimension {index.d} does not match expected {self.expected_dimension}"
                )
            if index.ntotal != len(chunk_ids):
                raise ValueError("FAISS vector count does not match chunk_ids.json length")
            self._index = index
            self._chunk_ids = tuple(chunk_ids)

    @staticmethod
    def _domain_chunk(record: dict[str, Any]) -> DocumentChunk:
        code = record.get("standard_code")
        standard_id = None
        if isinstance(code, str) and code.strip():
            try:
                standard_id = StandardId.parse(code)
            except ValueError:
                # Preserve a chunk with an unresolved identity; decision policy
                # can keep it ineligible without fabricating a standard code.
                standard_id = None
        return DocumentChunk(
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

    def search(self, query: str, limit: int) -> tuple[Candidate, ...]:
        """Return top dense candidates in FAISS rank order with cosine scores."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")

        with self._lock:
            self._ensure_loaded()
            if not self._chunk_ids:
                return ()

            import numpy as np

            vector = np.asarray(self.encoder.encode_text(query), dtype=np.float32).reshape(-1)
            if vector.size != self.expected_dimension or not np.isfinite(vector).all():
                raise ValueError("Encoder returned an invalid query vector")
            norm = float(np.linalg.norm(vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError("Encoder must return a unit-normalized query vector")

            result_count = min(limit, len(self._chunk_ids))
            scores, positions = self._index.search(vector.reshape(1, -1), result_count)
            hits: list[tuple[str, float]] = []
            for score, position in zip(scores[0], positions[0], strict=True):
                offset = int(position)
                if offset < 0:
                    continue
                if offset >= len(self._chunk_ids):
                    raise ValueError(f"FAISS returned an out-of-range vector position: {offset}")
                score_value = float(score)
                if not math.isfinite(score_value):
                    raise ValueError("FAISS returned a non-finite similarity score")
                hits.append((self._chunk_ids[offset], score_value))

            records = self.catalog.get_chunks_by_ids([chunk_id for chunk_id, _ in hits])
            missing_ids = [chunk_id for chunk_id, _ in hits if chunk_id not in records]
            if missing_ids:
                raise ValueError(
                    f"FAISS index references missing catalog chunks: {missing_ids[:5]}"
                )
            return tuple(
                Candidate(
                    chunk=self._domain_chunk(records[chunk_id]),
                    dense_score=score,
                )
                for chunk_id, score in hits
            )

    def close(self) -> None:
        """Release the native index object and its mapped data, if loaded."""
        with self._lock:
            self._index = None
            self._chunk_ids = None
            release_encoder = getattr(self.encoder, "release", None)
            if callable(release_encoder):
                release_encoder()
