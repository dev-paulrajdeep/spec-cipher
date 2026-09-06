"""Central configuration for retrieval construction and query defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration recorded with each persisted semantic index."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Verified on the repository's large SP 7 chunks.  Larger batches may be
    # selected explicitly by a deployment with sufficient memory.
    batch_size: int = 16
    normalize_embeddings: bool = True
    top_k: int = 5
    index_type: str = "IndexFlatIP"
    corpus_path: Path | None = None
    index_directory: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.index_type != "IndexFlatIP":
            raise ValueError("Only the exact IndexFlatIP strategy is currently supported")
        if not self.normalize_embeddings:
            raise ValueError("IndexFlatIP retrieval requires normalized embeddings for cosine similarity")
