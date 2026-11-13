"""Build a normalized-vector IndexFlatIP payload from chunk text records."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol


class BatchEncoder(Protocol):
    """Injected embedding adapter; model construction stays outside this module."""

    def encode_batch(self, texts: list[str]) -> Any: ...


class FaissBuildError(ValueError):
    """Raised when embeddings cannot produce a deterministic dense index."""


def build_dense_payloads(
    chunks: Iterable[Mapping[str, Any]],
    encoder: BatchEncoder,
    vectors_path: Path | str,
    chunk_ids_path: Path | str,
    *,
    dimension: int = 1024,
    batch_size: int = 32,
    faiss_module: Any | None = None,
) -> int:
    """Encode chunks in order and atomically create FAISS and chunk-ID payloads.

    The encoder is injected by the build command. This function loads FAISS and
    NumPy only when called, validates unit vectors, and never overwrites payloads.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    vector_target = Path(vectors_path)
    ids_target = Path(chunk_ids_path)
    if vector_target == ids_target:
        raise ValueError("vectors_path and chunk_ids_path must be different files")
    for target in (vector_target, ids_target):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing index payload: {target}")

    rows = list(chunks)
    chunk_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise FaissBuildError("Every chunk record must be a mapping")
        chunk_id = row.get("chunk_id")
        text = row.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise FaissBuildError("Every chunk must have a non-empty chunk_id")
        if not isinstance(text, str) or not text.strip():
            raise FaissBuildError(f"Chunk {chunk_id!r} must have non-empty text")
        chunk_ids.append(chunk_id)
        texts.append(text)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise FaissBuildError("Chunk IDs must be unique and preserve a one-to-one vector mapping")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to build the FAISS vector payload") from exc
    if faiss_module is None:
        try:
            import faiss as faiss_module
        except ImportError as exc:
            raise RuntimeError("FAISS is required to build the vector payload") from exc

    index = faiss_module.IndexFlatIP(dimension)
    if not rows:
        return _write_payloads(index, chunk_ids, vector_target, ids_target, faiss_module)

    for start in range(0, len(rows), batch_size):
        batch_texts = texts[start : start + batch_size]
        try:
            vectors = np.ascontiguousarray(encoder.encode_batch(batch_texts), dtype=np.float32)
        except Exception as exc:
            raise FaissBuildError(
                f"Embedding batch starting at chunk {start} failed: {exc}"
            ) from exc
        if vectors.shape != (len(batch_texts), dimension):
            raise FaissBuildError(
                f"Encoder returned shape {vectors.shape}; expected {(len(batch_texts), dimension)}"
            )
        if not np.isfinite(vectors).all():
            raise FaissBuildError(f"Encoder returned non-finite values at chunk batch {start}")
        norms = np.linalg.norm(vectors, axis=1)
        if not all(math.isclose(float(norm), 1.0, rel_tol=1e-3, abs_tol=1e-3) for norm in norms):
            raise FaissBuildError(f"Encoder returned non-unit vectors at chunk batch {start}")
        index.add(vectors)

    return _write_payloads(index, chunk_ids, vector_target, ids_target, faiss_module)


def _write_payloads(
    index: Any,
    chunk_ids: list[str],
    vector_target: Path,
    ids_target: Path,
    faiss_module: Any,
) -> int:
    vector_target.parent.mkdir(parents=True, exist_ok=True)
    ids_target.parent.mkdir(parents=True, exist_ok=True)
    temporary_vectors: Path | None = None
    temporary_ids: Path | None = None
    promoted: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{vector_target.name}.", suffix=".tmp", dir=vector_target.parent, delete=False
        ) as stream:
            temporary_vectors = Path(stream.name)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{ids_target.name}.",
            suffix=".tmp",
            dir=ids_target.parent,
            delete=False,
        ) as stream:
            temporary_ids = Path(stream.name)
            json.dump(chunk_ids, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        faiss_module.write_index(index, str(temporary_vectors))
        if not temporary_vectors.is_file() or temporary_vectors.stat().st_size == 0:
            raise FaissBuildError("FAISS did not write a non-empty index file")
        try:
            os.link(temporary_vectors, vector_target)
        except (OSError, PermissionError):
            if vector_target.exists():
                raise FileExistsError(f"Target already exists: {vector_target}")
            os.replace(temporary_vectors, vector_target)
        promoted.append(vector_target)

        try:
            os.link(temporary_ids, ids_target)
        except (OSError, PermissionError):
            if ids_target.exists():
                raise FileExistsError(f"Target already exists: {ids_target}")
            os.replace(temporary_ids, ids_target)
        promoted.append(ids_target)
        return len(chunk_ids)
    except OSError as exc:
        for target in promoted:
            target.unlink(missing_ok=True)
        raise FaissBuildError(f"Could not atomically create dense index payloads: {exc}") from exc
    finally:
        if temporary_vectors is not None:
            temporary_vectors.unlink(missing_ok=True)
        if temporary_ids is not None:
            temporary_ids.unlink(missing_ok=True)
