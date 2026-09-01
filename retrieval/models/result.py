"""Result model returned to the future intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass

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
