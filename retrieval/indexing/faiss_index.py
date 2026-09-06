"""Thin, explicit wrapper over a normalized-vector FAISS IndexFlatIP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FaissUnavailableError(RuntimeError):
    """Raised when the optional runtime FAISS dependency is unavailable."""


@dataclass(frozen=True)
class IndexSearchHit:
    """An explicit mapping from FAISS position to corpus chunk identity."""

    chunk_id: str
    similarity_score: float


class FaissIndex:
    """Exact cosine retrieval over unit-normalized vectors using ``IndexFlatIP``."""

    index_type = "IndexFlatIP"

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Index dimension must be positive")
        self._faiss = self._import_faiss()
        self._native = self._faiss.IndexFlatIP(dimension)
        self._chunk_ids: list[str] = []

    @classmethod
    def from_native(cls, native, chunk_ids: list[str]) -> "FaissIndex":
        """Wrap a deserialised index after persistence has validated its shape."""
        faiss = cls._import_faiss()
        if not isinstance(native, faiss.IndexFlatIP):
            raise ValueError("Loaded FAISS index must be an IndexFlatIP")
        if native.d < 1:
            raise ValueError("Loaded FAISS index has an invalid dimension")
        instance = cls.__new__(cls)
        instance._faiss = faiss
        instance._native = native
        instance._chunk_ids = list(chunk_ids)
        if native.ntotal != len(chunk_ids):
            raise ValueError("FAISS vector count does not match explicit chunk-ID mapping")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk-ID mapping contains duplicate values")
        return instance

    @property
    def dimension(self) -> int:
        return int(self._native.d)

    @property
    def count(self) -> int:
        return int(self._native.ntotal)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(self._chunk_ids)

    @property
    def native(self):
        """Underlying index for persistence only; callers should not mutate it."""
        return self._native

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        values = self._validated_vectors(vectors)
        if len(values) != len(chunk_ids):
            raise ValueError("Every vector must have exactly one explicit chunk ID")
        if not chunk_ids or any(not isinstance(value, str) or not value for value in chunk_ids):
            raise ValueError("At least one non-empty chunk ID is required")
        if len(chunk_ids) != len(set(chunk_ids)) or set(chunk_ids).intersection(self._chunk_ids):
            raise ValueError("Chunk IDs must be unique within an index")
        self._validate_unit_norms(values)
        self._native.add(values)
        self._chunk_ids.extend(chunk_ids)

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[IndexSearchHit]]:
        values = self._validated_vectors(query_vectors)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be positive")
        self._validate_unit_norms(values)
        if self.count == 0:
            return [[] for _ in range(len(values))]
        distances, positions = self._native.search(values, min(top_k, self.count))
        return [
            [
                IndexSearchHit(self._chunk_ids[int(position)], float(score))
                for score, position in zip(row_scores, row_positions, strict=True)
                if position >= 0
            ]
            for row_scores, row_positions in zip(distances, positions, strict=True)
        ]

    def _validated_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"Expected a two-dimensional array with dimension {self.dimension}")
        if values.shape[0] == 0:
            raise ValueError("At least one vector is required")
        if not np.isfinite(values).all():
            raise ValueError("Vectors must be finite")
        return np.ascontiguousarray(values)

    @staticmethod
    def _validate_unit_norms(values: np.ndarray) -> None:
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("IndexFlatIP cosine retrieval requires unit-normalized embeddings")

    @staticmethod
    def _import_faiss():
        try:
            import faiss
        except ImportError as exc:
            raise FaissUnavailableError("faiss-cpu is required for semantic indexing") from exc
        return faiss
