"""Result model returned to the future intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corpus.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk and its cosine similarity to the query.

    ``similarity_score`` is a semantic cosine similarity when the configured
    encoder normalizes embeddings. It is not applicability or confidence.
    """

    chunk: DocumentChunk
    similarity_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-ready retrieval contract for serving code.

        ``score`` is the same value as ``similarity_score``: cosine similarity
        between normalized query and chunk embeddings, not an applicability or
        confidence score.
        """
        return {
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
