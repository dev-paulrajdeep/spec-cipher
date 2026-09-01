"""Minimal extension boundary; no lexical engine or fusion policy is implemented here."""

from __future__ import annotations

from typing import Protocol

from retrieval.models import RetrievalResult
from retrieval.search.retriever import Retriever


class LexicalRetriever(Retriever, Protocol):
    """Future lexical retrievers implement the same output contract as semantic search."""


class HybridFusion(Protocol):
    """Future fusion policies combine semantic and lexical result lists explicitly."""

    def fuse(
        self,
        semantic_results: tuple[RetrievalResult, ...],
        lexical_results: tuple[RetrievalResult, ...],
        limit: int,
    ) -> tuple[RetrievalResult, ...]: ...
