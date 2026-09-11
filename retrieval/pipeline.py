"""Index construction/loading facade that hides encoder and FAISS implementation details."""

from __future__ import annotations

import logging
from time import perf_counter
from pathlib import Path

from corpus.models import DocumentChunk
from retrieval.config import RetrievalConfig
from retrieval.corpus_loader import CorpusChunkLoader, LoadedCorpus
from retrieval.embeddings import EmbeddingEncoder, SentenceTransformerEncoder
from retrieval.indexing import FaissIndex, IndexMetadata, IndexPersistence
from retrieval.indexing.persistence import corpus_fingerprint
from retrieval.search import SemanticRetriever
from retrieval.reranking import BGEReranker, CandidateReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Build, persist, reload, and query semantic corpus candidates."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        encoder: EmbeddingEncoder | None = None,
        loader: CorpusChunkLoader | None = None,
        persistence: IndexPersistence | None = None,
        reranker: CandidateReranker | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.encoder = encoder or SentenceTransformerEncoder(
            self.config.embedding_model,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            device=self.config.embedding_device,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
            max_seq_length=self.config.max_seq_length,
            embedding_dimension=self.config.embedding_dimension,
            use_fp16=self.config.embedding_use_fp16,
            model_version=self.config.embedding_model_version,
        )
        if self.encoder.model_name != self.config.embedding_model:
            raise ValueError("Encoder model_name must match RetrievalConfig.embedding_model")
        if self.encoder.normalize_embeddings != self.config.normalize_embeddings:
            raise ValueError("Encoder normalization must match RetrievalConfig")
        if self.config.embedding_dimension is not None and self.encoder.dimension != self.config.embedding_dimension:
            raise ValueError("Encoder dimension must match RetrievalConfig.embedding_dimension")
        self.loader = loader or CorpusChunkLoader()
        self.persistence = persistence or IndexPersistence()
        self.reranker = reranker or self._configured_reranker()

    def build_from_corpus(self, artifact_path: str | Path) -> SemanticRetriever:
        """Load canonical corpus JSON/JSONL then build an in-memory semantic index."""
        loaded = self.loader.load(artifact_path)
        return self.build(loaded.chunks, corpus_schema_version=loaded.schema_version)

    def build(self, chunks: tuple[DocumentChunk, ...], corpus_schema_version: str | None = None) -> SemanticRetriever:
        """Encode validated chunks once and create an exact normalized-vector FAISS index."""
        if not chunks:
            raise ValueError("Cannot build a retrieval index from an empty corpus")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Cannot build a retrieval index with duplicate chunk IDs")
        started = perf_counter()
        vectors = self.encoder.encode_batch([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape != (len(chunks), self.encoder.dimension):
            raise ValueError("Embedding encoder returned an incompatible batch shape")
        index = FaissIndex(self.encoder.dimension)
        index.add(vectors, ids)
        logger.info(
            "Built semantic index: chunks=%d model=%s dimension=%d index_type=%s duration_ms=%.2f",
            len(chunks), self.encoder.model_name, self.encoder.dimension, index.index_type, (perf_counter() - started) * 1000,
        )
        return SemanticRetriever(
            self.encoder, index, chunks, self.config.top_k,
            self.reranker, self.config.reranker_candidate_k,
            self.config.sequential_model_lifecycle,
        )

    def build_and_persist(self, artifact_path: str | Path, directory: str | Path) -> SemanticRetriever:
        """Build a corpus index and persist the index, mapping, chunks, and metadata bundle."""
        loaded = self.loader.load(artifact_path)
        retriever = self.build(loaded.chunks, loaded.schema_version)
        metadata = IndexMetadata.create(
            embedding_backend=getattr(self.encoder, "backend_name", "custom"),
            embedding_model=self.encoder.model_name,
            embedding_model_version=getattr(self.encoder, "model_version", "unknown"),
            embedding_dimension=self.encoder.dimension,
            normalize_embeddings=self.encoder.normalize_embeddings,
            corpus_schema_version=loaded.schema_version,
            chunks=loaded.chunks,
        )
        self.persistence.save(directory, retriever._index, loaded.chunks, metadata)
        return retriever

    def load(self, directory: str | Path) -> SemanticRetriever:
        """Load only an index compatible with this pipeline's embedding configuration."""
        bundle = self.persistence.load(
            directory,
            expected_embedding_backend=getattr(self.encoder, "backend_name", "custom"),
            expected_embedding_model=self.encoder.model_name,
            expected_embedding_model_version=getattr(self.encoder, "model_version", "unknown"),
            expected_dimension=self.encoder.dimension,
            expected_normalize_embeddings=self.encoder.normalize_embeddings,
        )
        if self.config.corpus_path is not None:
            loaded = self.loader.load(self.config.corpus_path)
            expected_fingerprint = corpus_fingerprint(loaded.chunks)
            if expected_fingerprint != bundle.metadata.corpus_fingerprint:
                raise IndexCompatibilityError(
                    "Persisted index corpus does not match configured corpus: "
                    f"{self.config.corpus_path}"
                )
        logger.info(
            "Loaded semantic index: chunks=%d model=%s dimension=%d index_type=%s",
            bundle.metadata.chunk_count, bundle.metadata.embedding_model,
            bundle.metadata.embedding_dimension, bundle.metadata.index_type,
        )
        return SemanticRetriever(
            self.encoder, bundle.index, bundle.chunks, self.config.top_k,
            self.reranker, self.config.reranker_candidate_k,
            self.config.sequential_model_lifecycle,
        )

    def _configured_reranker(self) -> CandidateReranker | None:
        if not self.config.reranker_enabled:
            return None
        return BGEReranker(
            self.config.reranker_model,
            model_version=self.config.reranker_model_version,
            device=self.config.reranker_device,
            batch_size=self.config.reranker_batch_size,
            max_seq_length=self.config.reranker_max_seq_length,
            allow_cpu_fallback=self.config.reranker_allow_cpu_fallback,
            use_fp16=self.config.reranker_use_fp16,
        )
