"""Central configuration for retrieval construction and query defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration recorded with each persisted semantic index."""

    embedding_model: str = "BAAI/bge-m3"
    embedding_model_version: str = "main"
    embedding_dimension: int | None = None
    embedding_use_fp16: bool = True
    embedding_device: str = "auto"
    allow_cpu_fallback: bool = False
    batch_size: int = 16
    max_seq_length: int | None = None
    normalize_embeddings: bool = True
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_version: str = "main"
    reranker_device: str = "auto"
    reranker_allow_cpu_fallback: bool = False
    reranker_batch_size: int = 16
    reranker_max_seq_length: int | None = None
    reranker_candidate_k: int | None = None
    reranker_use_fp16: bool = True
    top_k: int = 5
    index_type: str = "IndexFlatIP"
    corpus_path: Path | None = None
    index_directory: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if not isinstance(self.embedding_model_version, str) or not self.embedding_model_version.strip():
            raise ValueError("embedding_model_version must not be empty")
        if self.embedding_dimension is not None and (
            not isinstance(self.embedding_dimension, int)
            or isinstance(self.embedding_dimension, bool)
            or self.embedding_dimension < 1
        ):
            raise ValueError("embedding_dimension must be positive when provided")
        if not isinstance(self.embedding_use_fp16, bool):
            raise ValueError("embedding_use_fp16 must be a boolean")
        if self.embedding_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("embedding_device must be one of: auto, cuda, cpu")
        if not isinstance(self.allow_cpu_fallback, bool):
            raise ValueError("allow_cpu_fallback must be a boolean")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_seq_length is not None and (
            not isinstance(self.max_seq_length, int)
            or isinstance(self.max_seq_length, bool)
            or self.max_seq_length < 1
        ):
            raise ValueError("max_seq_length must be positive when provided")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.index_type != "IndexFlatIP":
            raise ValueError("Only the exact IndexFlatIP strategy is currently supported")
        if not self.normalize_embeddings:
            raise ValueError("IndexFlatIP retrieval requires normalized embeddings for cosine similarity")
        if not isinstance(self.reranker_enabled, bool):
            raise ValueError("reranker_enabled must be a boolean")
        if not isinstance(self.reranker_model, str) or not self.reranker_model.strip():
            raise ValueError("reranker_model must not be empty")
        if not isinstance(self.reranker_model_version, str) or not self.reranker_model_version.strip():
            raise ValueError("reranker_model_version must not be empty")
        if self.reranker_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("reranker_device must be one of: auto, cuda, cpu")
        if not isinstance(self.reranker_allow_cpu_fallback, bool):
            raise ValueError("reranker_allow_cpu_fallback must be a boolean")
        if not isinstance(self.reranker_use_fp16, bool):
            raise ValueError("reranker_use_fp16 must be a boolean")
        if not isinstance(self.reranker_batch_size, int) or isinstance(self.reranker_batch_size, bool) or self.reranker_batch_size < 1:
            raise ValueError("reranker_batch_size must be positive")
        if self.reranker_max_seq_length is not None and (
            not isinstance(self.reranker_max_seq_length, int)
            or isinstance(self.reranker_max_seq_length, bool)
            or self.reranker_max_seq_length < 1
        ):
            raise ValueError("reranker_max_seq_length must be positive when provided")
        if self.reranker_candidate_k is not None and (
            not isinstance(self.reranker_candidate_k, int)
            or isinstance(self.reranker_candidate_k, bool)
            or self.reranker_candidate_k < 1
        ):
            raise ValueError("reranker_candidate_k must be positive when provided")
