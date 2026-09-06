"""Corpus-backed semantic candidate retrieval for SpecCipher.

This package consumes corpus chunks. It intentionally contains no PDF parsing,
applicability assessment, reranking, or serving concerns.
"""

from .config import RetrievalConfig
from .pipeline import RetrievalPipeline
from .service import RetrievalService, RetrievalUnavailableError

__all__ = [
    "RetrievalConfig",
    "RetrievalPipeline",
    "RetrievalService",
    "RetrievalUnavailableError",
]
