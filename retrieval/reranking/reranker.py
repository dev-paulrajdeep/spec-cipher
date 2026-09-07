"""Replaceable contract for retrieval-stage candidate rerankers."""

from __future__ import annotations

from typing import Protocol

from retrieval.models import RetrievalResult


class CandidateReranker(Protocol):
    """Score and reorder only candidates already returned by FAISS."""

    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: tuple[RetrievalResult, ...],
        top_n: int | None = None,
    ) -> tuple[RetrievalResult, ...]: ...