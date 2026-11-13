from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speccipher.retrieval.dense.faiss_index import FaissVectorIndex
from speccipher.retrieval.lexical.fts5 import FTS5LexicalIndex


class FakeEncoder:
    dimension = 2

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def encode_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


class FakeCatalog:
    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        self.records = records
        self.calls: list[list[str]] = []

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        self.calls.append(chunk_ids)
        return {
            chunk_id: self.records[chunk_id] for chunk_id in chunk_ids if chunk_id in self.records
        }

    def search_fts(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        raise NotImplementedError


def _record(chunk_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "logical_doc_id": "pub-1",
        "standard_code": "IS 456:2000",
        "source_file_id": "source-1",
        "absolute_page_start": 10,
        "absolute_page_end": 11,
        "section_number": "5.1",
        "section_title": "Scope",
        "text": f"Evidence for {chunk_id}",
        "is_eligible": 1,
    }


def test_faiss_adapter_maps_positions_to_provenance_and_keeps_cosine_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "vectors.faiss"
    index_path.write_bytes(b"fake-faiss")
    chunk_ids_path = tmp_path / "chunk_ids.json"
    chunk_ids_path.write_text(json.dumps(["chunk-a", "chunk-b"]), encoding="utf-8")

    class FakeIndex:
        d = 2
        ntotal = 2

        def search(self, _query: np.ndarray, _limit: int) -> tuple[np.ndarray, np.ndarray]:
            return np.array([[0.91, 0.73]], dtype=np.float32), np.array([[1, 0]])

    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(read_index=lambda _path: FakeIndex()))
    encoder = FakeEncoder([1.0, 0.0])
    catalog = FakeCatalog({"chunk-a": _record("chunk-a"), "chunk-b": _record("chunk-b")})
    adapter = FaissVectorIndex(
        index_path,
        chunk_ids_path,
        encoder,
        catalog,
        expected_dimension=2,
    )

    candidates = adapter.search("concrete", limit=2)

    assert [candidate.chunk.chunk_id for candidate in candidates] == ["chunk-b", "chunk-a"]
    assert [candidate.dense_score for candidate in candidates] == pytest.approx([0.91, 0.73])
    assert all(candidate.lexical_score is None for candidate in candidates)
    assert candidates[0].chunk.standard_number == "IS 456:2000"
    assert candidates[0].chunk.absolute_page_start == 10
    assert encoder.calls == ["concrete"]
    assert catalog.calls == [["chunk-b", "chunk-a"]]


def test_faiss_adapter_fails_when_index_position_has_no_catalog_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "vectors.faiss").write_bytes(b"fake-faiss")
    (tmp_path / "chunk_ids.json").write_text(json.dumps(["missing"]), encoding="utf-8")

    class FakeIndex:
        d = 2
        ntotal = 1

        def search(self, _query: np.ndarray, _limit: int) -> tuple[np.ndarray, np.ndarray]:
            return np.array([[0.9]], dtype=np.float32), np.array([[0]])

    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(read_index=lambda _path: FakeIndex()))
    adapter = FaissVectorIndex(
        tmp_path / "vectors.faiss",
        tmp_path / "chunk_ids.json",
        FakeEncoder([1.0, 0.0]),
        FakeCatalog({}),
        expected_dimension=2,
    )

    with pytest.raises(ValueError, match="missing catalog chunks"):
        adapter.search("concrete", 1)


def test_faiss_adapter_rejects_non_unit_query_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "vectors.faiss").write_bytes(b"fake-faiss")
    (tmp_path / "chunk_ids.json").write_text('["chunk-a"]', encoding="utf-8")

    class EmptyIndex:
        d = 2
        ntotal = 1

    monkeypatch.setitem(
        sys.modules, "faiss", SimpleNamespace(read_index=lambda _path: EmptyIndex())
    )
    adapter = FaissVectorIndex(
        tmp_path / "vectors.faiss",
        tmp_path / "chunk_ids.json",
        FakeEncoder([2.0, 0.0]),
        FakeCatalog({}),
        expected_dimension=2,
    )

    with pytest.raises(ValueError, match="unit-normalized"):
        adapter.search("concrete", 1)


def test_fts_adapter_preserves_bm25_values_and_order() -> None:
    class SearchCatalog:
        def search_fts(self, _query: str, limit: int = 10) -> list[dict[str, object]]:
            assert limit == 2
            return [
                {**_record("a"), "rank_score": -0.8},
                {**_record("b"), "rank_score": -0.2},
            ]

    candidates = FTS5LexicalIndex(SearchCatalog()).search("concrete", limit=2)

    assert [candidate.chunk.chunk_id for candidate in candidates] == ["a", "b"]
    assert [candidate.lexical_score for candidate in candidates] == [-0.8, -0.2]
    assert all(candidate.dense_score is None for candidate in candidates)
