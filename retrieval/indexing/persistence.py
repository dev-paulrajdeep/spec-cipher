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

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.index_type != "IndexFlatIP":
            raise ValueError("Unsupported persisted index format")
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("Persisted embedding model is invalid")
        if (
            not isinstance(self.embedding_dimension, int)
            or isinstance(self.embedding_dimension, bool)
            or self.embedding_dimension < 1
        ):
            raise ValueError("Persisted embedding dimension is invalid")
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("Persisted normalization setting is invalid")
        if self.corpus_schema_version is not None and not isinstance(self.corpus_schema_version, str):
            raise ValueError("Persisted corpus schema version is invalid")
        if (
            not isinstance(self.corpus_fingerprint, str)
            or len(self.corpus_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.corpus_fingerprint)
        ):
            raise ValueError("Persisted corpus fingerprint is invalid")
        if not isinstance(self.chunk_count, int) or isinstance(self.chunk_count, bool) or self.chunk_count < 1:
            raise ValueError("Persisted chunk count is invalid")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.strip():
            raise ValueError("Persisted creation timestamp is invalid")

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
        chunks = self._chunks_from_rows(rows)
        if tuple(chunk.chunk_id for chunk in chunks) != tuple(chunk_ids):
            raise IndexCompatibilityError("Persisted chunks do not match the explicit ID mapping")
        if corpus_fingerprint(chunks) != metadata.corpus_fingerprint:
            raise IndexCompatibilityError("Persisted chunks do not match the recorded corpus fingerprint")
        faiss = FaissIndex._import_faiss()
        try:
            native = faiss.read_index(str(source / self.INDEX_FILE))
        except Exception as exc:
            raise IndexCompatibilityError("Persisted FAISS index could not be read") from exc
        try:
            index = FaissIndex.from_native(native, chunk_ids)
        except ValueError as exc:
            raise IndexCompatibilityError(f"Persisted FAISS index is incompatible: {exc}") from exc
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
        except OSError as exc:
            raise IndexCompatibilityError(f"Persisted JSON could not be read: {path.name}") from exc
        except json.JSONDecodeError as exc:
            raise IndexCompatibilityError(f"Invalid persisted JSON in {path.name}: {exc.msg}") from exc

    @staticmethod
    def _metadata_from_dict(value: object) -> IndexMetadata:
        if not isinstance(value, dict):
            raise IndexCompatibilityError("Persisted metadata is malformed")
        try:
            return IndexMetadata(**value)
        except (TypeError, ValueError) as exc:
            raise IndexCompatibilityError("Persisted metadata fields are invalid") from exc

    @staticmethod
    def _chunks_from_rows(rows: object) -> tuple[DocumentChunk, ...]:
        """Validate persisted chunks before they become provenance-bearing results."""
        if not isinstance(rows, list):
            raise IndexCompatibilityError("Persisted chunks are malformed")
        chunks: list[DocumentChunk] = []
        required = {
            "chunk_id", "standard_number", "source_file", "page_start", "page_end",
            "section_number", "section_title", "text",
        }
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or required.difference(row):
                raise IndexCompatibilityError(f"Persisted chunk {position} is malformed")
            chunk_id = row["chunk_id"]
            source_file = row["source_file"]
            text = row["text"]
            standard_number = row["standard_number"]
            if (
                not isinstance(chunk_id, str) or not chunk_id.strip()
                or not isinstance(source_file, str) or not source_file.strip()
                or not isinstance(text, str) or not text.strip()
                or (standard_number is not None and (not isinstance(standard_number, str) or not standard_number.strip()))
            ):
                raise IndexCompatibilityError(f"Persisted chunk {position} has invalid provenance or text")
            page_start = row["page_start"]
            page_end = row["page_end"]
            if (
                not isinstance(page_start, int) or isinstance(page_start, bool)
                or not isinstance(page_end, int) or isinstance(page_end, bool)
                or page_start < 1 or page_end < page_start
            ):
                raise IndexCompatibilityError(f"Persisted chunk {position} has an invalid page range")
            section_number = row["section_number"]
            section_title = row["section_title"]
            if any(value is not None and not isinstance(value, str) for value in (section_number, section_title)):
                raise IndexCompatibilityError(f"Persisted chunk {position} has invalid section metadata")
            element_ids = row.get("element_ids", [])
            if (
                not isinstance(element_ids, list)
                or any(not isinstance(value, str) or not value.strip() for value in element_ids)
            ):
                raise IndexCompatibilityError(f"Persisted chunk {position} has invalid element provenance")
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                standard_number=standard_number,
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                section_number=section_number,
                section_title=section_title,
                text=text,
                element_ids=tuple(element_ids),
            ))
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise IndexCompatibilityError("Persisted chunks contain duplicate chunk IDs")
        return tuple(chunks)

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
