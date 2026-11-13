"""Reranker port and standard-library-only deterministic fallback."""

from __future__ import annotations

from typing import Protocol

from speccipher.domain.candidates import Candidate, RerankScore
from speccipher.domain.requirements import tokenize


class CandidateReranker(Protocol):
    """Score each supplied candidate once, retaining its stable chunk ID."""

    @property
    def strategy(self) -> str: ...

    def score(
        self, requirement: str, candidates: tuple[Candidate, ...]
    ) -> dict[str, RerankScore]: ...

    def close(self) -> None: ...


class TokenOverlapReranker:
    """Transparent Jaccard fallback that works without model weights or network."""

    strategy = "token_overlap"

    def score(self, requirement: str, candidates: tuple[Candidate, ...]) -> dict[str, RerankScore]:
        query_terms = tokenize(requirement)
        scores: dict[str, RerankScore] = {}
        for candidate in candidates:
            text_terms = tokenize(candidate.chunk.text)
            union = query_terms | text_terms
            score = len(query_terms & text_terms) / len(union) if union else 0.0
            scores[candidate.chunk.chunk_id] = RerankScore(score, score)
        return scores

    def close(self) -> None:
        """The token-overlap reranker owns no external resources."""
