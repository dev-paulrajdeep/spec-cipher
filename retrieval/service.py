"""Stable application-facing retrieval facade.

Serving code imports this module rather than constructing FAISS indexes or
embedding encoders directly.  Index construction remains an offline concern
of :class:`retrieval.pipeline.RetrievalPipeline`.
"""

from __future__ import annotations

from pathlib import Path

from .config import RetrievalConfig
from .models import RetrievalResult
from .pipeline import RetrievalPipeline
from .search import SemanticRetriever


class RetrievalUnavailableError(RuntimeError):
    """Raised when ``search`` is called before a persisted index is loaded."""


class RetrievalService:
    """The stable boundary between serving code and the retrieval implementation.

    The only query operation exposed to application callers is ``search``.
    It returns corpus-backed :class:`RetrievalResult` objects and deliberately
    does not expose embedding or FAISS implementation details.
    """

    def __init__(self, _pipeline: RetrievalPipeline | None = None) -> None:
        self._pipeline = _pipeline or RetrievalPipeline()
        self._retriever: SemanticRetriever | None = None

    @classmethod
    def from_persisted(
        cls,
        directory: str | Path,
        *,
        config: RetrievalConfig | None = None,
    ) -> "RetrievalService":
        """Load a compatible persisted index bundle and return a ready service."""
        service = cls(_pipeline=RetrievalPipeline(config))
        service.load(directory)
        return service

    def load(self, directory: str | Path) -> None:
        """Load a persisted index bundle after model/configuration validation."""
        self._retriever = self._pipeline.load(directory)

    def search(self, query: str, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Return up to ``top_k`` provenance-preserving semantic candidates."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be positive")
        if self._retriever is None:
            raise RetrievalUnavailableError(
                "Retrieval service is not loaded; call load() or from_persisted() first"
            )
        return self._retriever.search(query, top_k=top_k)
