"""Future-proof protocol shared by semantic and eventual lexical retrievers."""

from __future__ import annotations

from typing import Protocol

from retrieval.models import RetrievalResult


class Retriever(Protocol):
    """Candidate retrieval contract consumed by the intelligence layer later."""

    def search(self, query: str, top_k: int | None = None) -> tuple[RetrievalResult, ...]: ...
