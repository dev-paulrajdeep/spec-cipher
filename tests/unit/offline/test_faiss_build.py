from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speccipher.offline.indexing.faiss_build import FaissBuildError, build_dense_payloads


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors: list[list[float]] = []

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.extend(vectors.tolist())


class FakeFaiss:
    def IndexFlatIP(self, dimension: int) -> FakeIndex:
        self.index = FakeIndex(dimension)
        return self.index

    def write_index(self, index: FakeIndex, path: str) -> None:
        Path(path).write_text(json.dumps(index.vectors), encoding="utf-8")


class FakeEncoder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        self.batches.append(texts)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def test_build_dense_payloads_preserves_ids_and_batches(tmp_path):
    encoder = FakeEncoder()
    faiss = FakeFaiss()
    vectors = tmp_path / "vectors.faiss"
    ids = tmp_path / "chunk_ids.json"

    count = build_dense_payloads(
        [
            {"chunk_id": "c1", "text": "first"},
            {"chunk_id": "c2", "text": "second"},
            {"chunk_id": "c3", "text": "third"},
        ],
        encoder,
        vectors,
        ids,
        dimension=2,
        batch_size=2,
        faiss_module=faiss,
    )

    assert count == 3
    assert encoder.batches == [["first", "second"], ["third"]]
    assert json.loads(ids.read_text(encoding="utf-8")) == ["c1", "c2", "c3"]
    assert len(json.loads(vectors.read_text(encoding="utf-8"))) == 3


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.array([[2.0, 0.0]], dtype=np.float32), "non-unit"),
        (np.array([[np.nan, 0.0]], dtype=np.float32), "non-finite"),
        (np.array([[1.0, 0.0, 0.0]], dtype=np.float32), "expected"),
    ],
)
def test_builder_rejects_invalid_vectors_without_creating_payloads(tmp_path, vectors, message):
    class InvalidEncoder:
        def encode_batch(self, _texts: list[str]) -> np.ndarray:
            return vectors

    vector_path = tmp_path / "vectors.faiss"
    ids_path = tmp_path / "chunk_ids.json"
    with pytest.raises(FaissBuildError, match=message):
        build_dense_payloads(
            [{"chunk_id": "c1", "text": "text"}],
            InvalidEncoder(),
            vector_path,
            ids_path,
            dimension=2,
            faiss_module=FakeFaiss(),
        )
    assert not vector_path.exists()
    assert not ids_path.exists()


def test_builder_rejects_duplicate_ids_and_preserves_existing_payload(tmp_path):
    existing = tmp_path / "vectors.faiss"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        build_dense_payloads(
            [{"chunk_id": "c1", "text": "first"}, {"chunk_id": "c1", "text": "again"}],
            FakeEncoder(),
            existing,
            tmp_path / "chunk_ids.json",
            dimension=2,
            faiss_module=FakeFaiss(),
        )
    assert existing.read_bytes() == b"keep"

    with pytest.raises(FaissBuildError, match="unique"):
        build_dense_payloads(
            [{"chunk_id": "c1", "text": "first"}, {"chunk_id": "c1", "text": "again"}],
            FakeEncoder(),
            tmp_path / "other.faiss",
            tmp_path / "other_ids.json",
            dimension=2,
            faiss_module=FakeFaiss(),
        )


def test_builder_writes_empty_flat_index_without_encoder_calls(tmp_path):
    encoder = FakeEncoder()
    count = build_dense_payloads(
        [],
        encoder,
        tmp_path / "vectors.faiss",
        tmp_path / "chunk_ids.json",
        dimension=2,
        faiss_module=FakeFaiss(),
    )

    assert count == 0
    assert encoder.batches == []
    assert json.loads((tmp_path / "chunk_ids.json").read_text(encoding="utf-8")) == []
