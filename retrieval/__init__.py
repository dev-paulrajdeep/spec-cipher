"""Corpus-backed semantic candidate retrieval for SpecCipher.

This package consumes corpus chunks. It intentionally contains no PDF parsing,
applicability assessment, reranking, or serving concerns.
"""

from .config import RetrievalConfig
from .pipeline import RetrievalPipeline

__all__ = ["RetrievalConfig", "RetrievalPipeline"]
