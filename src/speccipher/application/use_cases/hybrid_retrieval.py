"""Hybrid dense and lexical retrieval with deterministic Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite

from speccipher.application.ports.search import LexicalIndex, VectorIndex
from speccipher.domain.candidates import Candidate

DEFAULT_RRF_K = 60


def _validate_limit(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _prepare_ranked_list(
    candidates: Sequence[Candidate], *, score_field: str
) -> tuple[tuple[int, Candidate], ...]:
    prepared: dict[str, tuple[int, Candidate]] = {}
    for rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Candidate):
            raise TypeError("search ports must return Candidate objects")
        score = getattr(candidate, score_field)
        if score is None or not isfinite(score):
            raise ValueError(
                f"candidate {candidate.chunk.chunk_id!r} requires a finite {score_field}"
            )
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in prepared:
            previous = prepared[chunk_id][1]
            if previous.chunk != candidate.chunk:
                raise ValueError(f"candidate chunk_id {chunk_id!r} refers to conflicting chunks")
            # A repeated row in an index is counted only at its first rank.
            continue
        if score_field == "dense_score":
            prepared[chunk_id] = (
                rank,
                replace(candidate, lexical_score=None, rrf_score=0.0, rerank_score=None),
            )
        else:
            prepared[chunk_id] = (
                rank,
                replace(candidate, dense_score=None, rrf_score=0.0, rerank_score=None),
            )
    return tuple(prepared.values())


def fuse_ranked_candidates(
    dense_candidates: Sequence[Candidate],
    lexical_candidates: Sequence[Candidate],
    *,
    k: int = DEFAULT_RRF_K,
) -> tuple[Candidate, ...]:
    """Fuse independently ranked search results while retaining each score type.

    Rank positions come from the order of each input sequence. Duplicate chunk IDs
    within one sequence count once at their first rank. If the same ID resolves to
    different provenance/content across indexes, the use case fails rather than
    silently choosing one representation.
    """
    _validate_limit("k", k)
    dense = _prepare_ranked_list(dense_candidates, score_field="dense_score")
    lexical = _prepare_ranked_list(lexical_candidates, score_field="lexical_score")

    merged: dict[str, Candidate] = {}
    scores: dict[str, float] = {}
    for rank, candidate in dense:
        chunk_id = candidate.chunk.chunk_id
        merged[chunk_id] = candidate
        scores[chunk_id] = 1.0 / (k + rank)

    for rank, candidate in lexical:
        chunk_id = candidate.chunk.chunk_id
        prior = merged.get(chunk_id)
        if prior is None:
            merged[chunk_id] = candidate
            scores[chunk_id] = 0.0
        elif prior.chunk != candidate.chunk:
            raise ValueError(f"candidate chunk_id {chunk_id!r} refers to conflicting chunks")
        merged[chunk_id] = replace(merged[chunk_id], lexical_score=candidate.lexical_score)
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return tuple(replace(merged[chunk_id], rrf_score=scores[chunk_id]) for chunk_id in ordered_ids)


@dataclass(frozen=True)
class HybridRetrieval:
    """Application boundary that queries dense and lexical ports then applies RRF."""

    vector_index: VectorIndex
    lexical_index: LexicalIndex
    rrf_k: int = DEFAULT_RRF_K

    def __post_init__(self) -> None:
        _validate_limit("rrf_k", self.rrf_k)

    def retrieve(self, query: str, *, limit: int = 50) -> tuple[Candidate, ...]:
        """Return deterministic fused candidates for a non-empty requirement query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        _validate_limit("limit", limit)
        dense = self.vector_index.search(query, limit)
        lexical = self.lexical_index.search(query, limit)
        return fuse_ranked_candidates(dense, lexical, k=self.rrf_k)

    def close(self) -> None:
        """Release adapter resources when they expose an explicit close hook."""
        close = getattr(self.vector_index, "close", None)
        if callable(close):
            close()
