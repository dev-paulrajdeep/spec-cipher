"""Retrieval Engine Layer (Layer 4 Sibling).

Provides hybrid retrieval (dense FAISS vector search, lexical SQLite FTS5 search),
Reciprocal Rank Fusion (RRF), single-stage reranking, and the stable RetrievalService facade.

INVARIANT: Sibling engines in Layer 4 must NEVER import each other!
"""

from __future__ import annotations

__all__: list[str] = []
