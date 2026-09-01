"""Persist FAISS, explicit mapping, provenance, and compatibility metadata together."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from corpus.models import DocumentChunk
from .faiss_index import FaissIndex


class IndexCompatibilityError(RuntimeError):
    """Raised rather than silently searching with a mismatched index/model."""


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: str
    index_type: str
    embedding_model: str
    embedding_dimension: int
    normalize_embeddings: bool
    corpus_schema_version: str | None
    corpus_fingerprint: str
    chunk_count: int
    created_at_utc: str

    @classmethod
    def create(
        cls,
        *,
        embedding_model: str,
        embedding_dimension: int,
        normalize_embeddings: bool,
        corpus_schema_version: str | None,
        chunks: tuple[DocumentChunk, ...],
    ) -> "IndexMetadata":
        return cls(
            schema_version="1.0",
            index_type="IndexFlatIP",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            normalize_embeddings=normalize_embeddings,
            corpus_schema_version=corpus_schema_version,
            corpus_fingerprint=corpus_fingerprint(chunks),
            chunk_count=len(chunks),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class PersistedIndex:
    index: FaissIndex
    chunks: tuple[DocumentChunk, ...]
    metadata: IndexMetadata


def corpus_fingerprint(chunks: tuple[DocumentChunk, ...]) -> str:
    """Stable fingerprint used to identify exactly which corpus was indexed."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class IndexPersistence:
    """Store a complete, reloadable index bundle in one directory."""

    INDEX_FILE = "index.faiss"
    MAPPING_FILE = "chunk_ids.json"
    CHUNKS_FILE = "chunks.json"
    METADATA_FILE = "metadata.json"

    def save(self, directory: str | Path, index: FaissIndex, chunks: tuple[DocumentChunk, ...], metadata: IndexMetadata) -> None:
        target = Path(directory)
        if index.count != len(chunks) or index.count != metadata.chunk_count:
            raise ValueError("Index, chunk records, and metadata must have identical counts")
        if tuple(chunk.chunk_id for chunk in chunks) != index.chunk_ids:
            raise ValueError("Chunk records must be ordered by the explicit index ID mapping")
        if corpus_fingerprint(chunks) != metadata.corpus_fingerprint:
            raise ValueError("Metadata corpus fingerprint does not match chunks")
        target.mkdir(parents=True, exist_ok=True)
        index._faiss.write_index(index.native, str(target / self.INDEX_FILE))
        self._write_json(target / self.MAPPING_FILE, list(index.chunk_ids))
        self._write_json(target / self.CHUNKS_FILE, [asdict(chunk) for chunk in chunks])
        self._write_json(target / self.METADATA_FILE, asdict(metadata))

    def load(
        self,
        directory: str | Path,
        *,
        expected_embedding_model: str,
        expected_dimension: int,
        expected_normalize_embeddings: bool,
    ) -> PersistedIndex:
        source = Path(directory)
        required = [source / name for name in (self.INDEX_FILE, self.MAPPING_FILE, self.CHUNKS_FILE, self.METADATA_FILE)]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Persisted index is incomplete; missing: {', '.join(missing)}")
        metadata = self._metadata_from_dict(self._read_json(source / self.METADATA_FILE))
        self._validate_compatibility(metadata, expected_embedding_model, expected_dimension, expected_normalize_embeddings)
        chunk_ids = self._read_json(source / self.MAPPING_FILE)
        rows = self._read_json(source / self.CHUNKS_FILE)
        if not isinstance(chunk_ids, list) or not all(isinstance(item, str) for item in chunk_ids):
            raise IndexCompatibilityError("Persisted chunk-ID mapping is malformed")
        if not isinstance(rows, list):
            raise IndexCompatibilityError("Persisted chunks are malformed")
        chunks = tuple(DocumentChunk(**row) for row in rows)
        if tuple(chunk.chunk_id for chunk in chunks) != tuple(chunk_ids):
            raise IndexCompatibilityError("Persisted chunks do not match the explicit ID mapping")
        if corpus_fingerprint(chunks) != metadata.corpus_fingerprint:
            raise IndexCompatibilityError("Persisted chunks do not match the recorded corpus fingerprint")
        faiss = FaissIndex._import_faiss()
        native = faiss.read_index(str(source / self.INDEX_FILE))
        index = FaissIndex.from_native(native, chunk_ids)
        if index.dimension != metadata.embedding_dimension or index.count != metadata.chunk_count:
            raise IndexCompatibilityError("Persisted FAISS index does not match its metadata")
        return PersistedIndex(index=index, chunks=chunks, metadata=metadata)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IndexCompatibilityError(f"Invalid persisted JSON in {path.name}: {exc.msg}") from exc

    @staticmethod
    def _metadata_from_dict(value: object) -> IndexMetadata:
        if not isinstance(value, dict):
            raise IndexCompatibilityError("Persisted metadata is malformed")
        try:
            return IndexMetadata(**value)
        except TypeError as exc:
            raise IndexCompatibilityError("Persisted metadata fields are invalid") from exc

    @staticmethod
    def _validate_compatibility(metadata: IndexMetadata, model: str, dimension: int, normalized: bool) -> None:
        if metadata.schema_version != "1.0" or metadata.index_type != "IndexFlatIP":
            raise IndexCompatibilityError("Unsupported persisted index format")
        if metadata.embedding_model != model:
            raise IndexCompatibilityError("Persisted index was built with a different embedding model")
        if metadata.embedding_dimension != dimension:
            raise IndexCompatibilityError("Persisted index has an incompatible embedding dimension")
        if metadata.normalize_embeddings != normalized:
            raise IndexCompatibilityError("Persisted index has an incompatible normalization strategy")
