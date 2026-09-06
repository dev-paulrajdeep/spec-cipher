"""Query encoding and FAISS retrieval, with all corpus provenance retained."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from corpus.models import DocumentChunk
from retrieval.embeddings import EmbeddingEncoder
from retrieval.indexing import FaissIndex
from retrieval.models import RetrievalResult, StandardCandidate, group_results_by_standard

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic Top-K candidate generator; scores are not recommendations."""

    def __init__(self, encoder: EmbeddingEncoder, index: FaissIndex, chunks: tuple[DocumentChunk, ...], default_top_k: int = 5) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be positive")
        if index.dimension != encoder.dimension:
            raise ValueError("Embedding encoder dimension does not match the FAISS index")
        if not encoder.normalize_embeddings:
            raise ValueError("SemanticRetriever requires normalized embeddings for cosine similarity")
        if tuple(chunk.chunk_id for chunk in chunks) != index.chunk_ids:
            raise ValueError("Chunks must align exactly with the index's explicit chunk-ID mapping")
        self._encoder = encoder
        self._index = index
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._default_top_k = default_top_k

    def search(self, query: str, top_k: int | None = None) -> tuple[RetrievalResult, ...]:
        """Return at most Top-K chunks ordered by decreasing cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        requested = self._default_top_k if top_k is None else top_k
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            raise ValueError("top_k must be positive")
        started = perf_counter()
        vector = np.asarray(self._encoder.encode_text(query), dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._index.dimension:
            raise ValueError("Query encoder returned an incompatible embedding dimension")
        hits = self._index.search(vector.reshape(1, -1), requested)[0]
        results = tuple(
            RetrievalResult(chunk=self._chunks[hit.chunk_id], similarity_score=hit.similarity_score, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        )
        logger.info("Semantic query completed: candidates=%d duration_ms=%.2f", len(results), (perf_counter() - started) * 1000)
        return results

    def search_candidates(self, query: str, top_k: int | None = None) -> tuple[StandardCandidate, ...]:
        """Provide a standard grouping while retaining every returned chunk as evidence."""
        return group_results_by_standard(self.search(query, top_k))
