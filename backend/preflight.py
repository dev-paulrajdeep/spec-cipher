"""Actionable startup validation for the SpecCipher serving bundle."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from retrieval.config import RetrievalConfig
from retrieval.corpus_config import resolve_retrieval_corpus_path
from retrieval.corpus_loader import CorpusChunkLoader
from retrieval.indexing import FaissIndex
from retrieval.indexing.persistence import IndexPersistence, corpus_fingerprint


class PreflightError(RuntimeError):
    """Raised when a serving prerequisite is missing or inconsistent."""


def validate_bundle(
    index_directory: str | Path,
    corpus_path: str | Path | None = None,
    *,
    check_ollama: bool = False,
    ollama_host: str = "http://localhost:11434",
) -> None:
    """Validate static bundle identity and optionally the Ollama endpoint."""
    index_directory = Path(index_directory)
    corpus_path = Path(corpus_path) if corpus_path is not None else resolve_retrieval_corpus_path()
    required_names = (
        IndexPersistence.INDEX_FILE,
        IndexPersistence.MAPPING_FILE,
        IndexPersistence.CHUNKS_FILE,
        IndexPersistence.METADATA_FILE,
    )
    missing = [name for name in required_names if not (index_directory / name).is_file()]
    if missing:
        raise PreflightError(
            f"Retrieval bundle is incomplete at {index_directory}; missing: {', '.join(missing)}"
        )
    if not corpus_path.is_file():
        raise PreflightError(f"Configured corpus artifact is missing: {corpus_path}")

    try:
        metadata = json.loads((index_directory / IndexPersistence.METADATA_FILE).read_text(encoding="utf-8"))
        loaded = CorpusChunkLoader().load(corpus_path)
        persisted_rows = json.loads((index_directory / IndexPersistence.CHUNKS_FILE).read_text(encoding="utf-8"))
        persisted_chunks = IndexPersistence._chunks_from_rows(persisted_rows)
        mapping = json.loads((index_directory / IndexPersistence.MAPPING_FILE).read_text(encoding="utf-8"))
        expected = corpus_fingerprint(loaded.chunks)
        persisted_fingerprint = corpus_fingerprint(persisted_chunks)
        actual = metadata.get("corpus_fingerprint") if isinstance(metadata, dict) else None
        index_chunk_count = metadata.get("chunk_count") if isinstance(metadata, dict) else None
        embedding_dimension = metadata.get("embedding_dimension") if isinstance(metadata, dict) else None
        native = FaissIndex._import_faiss().read_index(str(index_directory / IndexPersistence.INDEX_FILE))
        vector_count = int(native.ntotal)
        vector_dimension = int(native.d)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise PreflightError(f"Could not validate corpus/index metadata: {exc}") from exc
    if actual != expected:
        raise PreflightError(
            "Retrieval index does not match the configured corpus. "
            "Rebuild with: python scripts/update_corpus.py"
        )
    if persisted_fingerprint != expected:
        raise PreflightError("Persisted index chunks do not match the configured corpus")
    if not isinstance(mapping, list) or tuple(mapping) != tuple(chunk.chunk_id for chunk in persisted_chunks):
        raise PreflightError("Persisted index chunk mapping does not match its chunk payload")
    if index_chunk_count != len(loaded.chunks):
        raise PreflightError(
            f"Corpus/index chunk count mismatch: corpus={len(loaded.chunks)} index_metadata={index_chunk_count}"
        )
    if not isinstance(embedding_dimension, int) or embedding_dimension < 1:
        raise PreflightError("Persisted embedding dimension is missing or invalid")
    if vector_count != len(loaded.chunks) or vector_count != index_chunk_count:
        raise PreflightError(
            f"Corpus/index vector count mismatch: corpus={len(loaded.chunks)} vectors={vector_count} metadata={index_chunk_count}"
        )
    if vector_count == 0:
        raise PreflightError(
            "The retrieval index contains no recommendation-eligible chunks; "
            "the service cannot be marked ready. Add an identity-resolved source PDF."
        )
    if vector_dimension != embedding_dimension:
        raise PreflightError(
            f"Persisted FAISS dimension mismatch: vectors={vector_dimension} metadata={embedding_dimension}"
        )
    if check_ollama:
        parsed = urlsplit(ollama_host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise PreflightError(f"Ollama host is invalid: {ollama_host}")
        try:
            with urlopen(f"{ollama_host.rstrip('/')}/api/tags", timeout=2) as response:
                if response.status >= 400:
                    raise PreflightError(f"Ollama health check returned HTTP {response.status}")
        except PreflightError:
            raise
        except OSError as exc:
            raise PreflightError(
                f"Ollama is unavailable at {ollama_host}; explanations will use the deterministic fallback: {exc}"
            ) from exc


def run_preflight(index_directory: str | Path, corpus_path: str | Path | None = None) -> None:
    """CLI-friendly preflight that does not load the embedding model."""
    validate_bundle(index_directory, corpus_path)
    print("Preflight OK: retrieval bundle and corpus identity are valid")
