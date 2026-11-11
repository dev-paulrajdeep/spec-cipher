"""Reciprocal Rank Fusion (RRF) for hybrid retrieval (Layer 4 Retrieval)."""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_candidates: list[dict[str, Any]],
    lexical_candidates: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge dense vector and lexical rankings using Reciprocal Rank Fusion (RRF).

    Formula: score(c) = sum(1 / (k + rank_i))
    Tie-breaking: Deterministically by combined score descending, then chunk_id ascending.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for rank, cand in enumerate(dense_candidates, 1):
        cid = cand["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        items[cid] = cand

    for rank, cand in enumerate(lexical_candidates, 1):
        cid = cand["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in items:
            items[cid] = cand

    # Deterministic tie-breaking: score descending, then chunk_id ascending
    sorted_cids = sorted(scores.keys(), key=lambda c: (-scores[c], c))
    return [dict(items[c], rrf_score=scores[c]) for c in sorted_cids]
