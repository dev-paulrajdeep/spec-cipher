"""Result model returned to the future intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corpus.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk with semantic and optional second-stage scores.

    ``similarity_score`` is the first-stage BGE-M3/FAISS cosine similarity.
    ``reranker_score`` is optional and never replaces that semantic score.
    """

    chunk: DocumentChunk
    similarity_score: float
    rank: int
    reranker_score: float | None = None

    @property
    def semantic_score(self) -> float:
        """Explicit name for the original first-stage semantic score."""
        return self.similarity_score

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-ready retrieval contract for serving code.

        ``score`` is the same value as ``similarity_score``: cosine similarity
        between normalized query and chunk embeddings, not an applicability or
        confidence score.
        """
        result = {
            "standard_number": self.chunk.standard_number,
            "title": None,
            "section_number": self.chunk.section_number,
            "section_title": self.chunk.section_title,
            "text": self.chunk.text,
            "score": self.similarity_score,
            "rank": self.rank,
            "source": {
                "chunk_id": self.chunk.chunk_id,
                "source_file": self.chunk.source_file,
                "page_start": self.chunk.page_start,
                "page_end": self.chunk.page_end,
                "element_ids": list(self.chunk.element_ids),
            },
        }
        if self.reranker_score is not None:
            result["semantic_score"] = self.semantic_score
            result["reranker_score"] = self.reranker_score
        return result
