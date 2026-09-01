"""Deterministic default reranking and the common reranker contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intelligence.models import RerankResult
from intelligence.text import tokenize
from retrieval.models import RetrievalResult


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: RetrievalResult
    result: RerankResult


class Reranker(Protocol):
    def rerank(self, requirement: str, candidates: tuple[RetrievalResult, ...]) -> tuple[RerankedCandidate, ...]: ...


class TokenOverlapReranker:
    """Transparent fallback based on Jaccard overlap of requirement and chunk terms."""

    strategy = "token_overlap"

    def rerank(self, requirement: str, candidates: tuple[RetrievalResult, ...]) -> tuple[RerankedCandidate, ...]:
        query_terms = tokenize(requirement)
        scored = [
            (self._score(query_terms, tokenize(candidate.chunk.text)), original_rank, candidate)
            for original_rank, candidate in enumerate(candidates, start=1)
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk.chunk_id))
        return tuple(
            RerankedCandidate(candidate, RerankResult(score, score, rank, self.strategy))
            for rank, (score, _, candidate) in enumerate(scored, start=1)
        )

    @staticmethod
    def _score(query_terms: frozenset[str], text_terms: frozenset[str]) -> float:
        union = query_terms.union(text_terms)
        return len(query_terms.intersection(text_terms)) / len(union) if union else 0.0
