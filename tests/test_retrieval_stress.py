from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from corpus.models import DocumentChunk
from retrieval.corpus_loader import CorpusChunkLoadError, CorpusChunkLoader
from retrieval.indexing import IndexCompatibilityError
from retrieval.pipeline import RetrievalPipeline
from retrieval.config import RetrievalConfig


class StressEncoder:
    model_name = "stress-encoder"
    dimension = 3
    normalize_embeddings = True

    def encode_text(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.tile(self.encode_text("any"), (len(texts), 1))


def make_chunk(identifier: str, *, element_ids: tuple[str, ...] = ()) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        standard_number="IS/7:2026 [special]",
        source_file="source ü.pdf",
        page_start=2,
        page_end=4,
        section_number=None,
        section_title="Scope — long title",
        text="Concrete and water resistance requirements.",
        element_ids=element_ids,
    )


def make_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline(
        RetrievalConfig(embedding_model="stress-encoder", top_k=3),
        encoder=StressEncoder(),
    )


def to_row(item: DocumentChunk) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "standard_number": item.standard_number,
        "source_file": item.source_file,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_number": item.section_number,
        "section_title": item.section_title,
        "text": item.text,
        "element_ids": list(item.element_ids),
    }


def test_fuzzed_queries_are_controlled_repeatable_and_top_k_bounded() -> None:
    chunks = (make_chunk("id:one"), make_chunk("id/ two"), make_chunk("id__three"))
    retriever = make_pipeline().build(chunks)

    invalid_queries = (None, 7, [], {}, b"bytes", "", " \t\n")
    for query in invalid_queries:
        with pytest.raises(ValueError, match="non-empty string"):
            retriever.search(query)  # type: ignore[arg-type]

    invalid_top_k = (0, -1, 1.5, "3", True)
    for top_k in invalid_top_k:
        with pytest.raises(ValueError, match="top_k"):
            retriever.search("valid query", top_k)  # type: ignore[arg-type]

    valid_queries = (
        "!@#$%^&*()",
        "ÜBER concrete — 2026",
        "CONCRETE concrete concrete " * 1000,
    )
    for query in valid_queries:
        results = retriever.search(query, top_k=100000)
        assert len(results) == len(chunks)
        assert all(np.isfinite(result.similarity_score) for result in results)
        assert all(results[index].similarity_score >= results[index + 1].similarity_score for index in range(len(results) - 1))
        assert retriever.search(query, top_k=2) == retriever.search(query, top_k=2)


def test_repeated_concurrent_reads_do_not_leak_state() -> None:
    retriever = make_pipeline().build(tuple(make_chunk(f"chunk-{index}") for index in range(8)))

    def search(_: int):
        return retriever.search("same query", top_k=5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(search, range(64)))

    assert all(result == results[0] for result in results)
    assert [item.rank for item in results[0]] == [1, 2, 3, 4, 5]


def test_unusual_provenance_survives_build_and_persistence(tmp_path) -> None:
    chunks = (make_chunk("id:one", element_ids=("element/1", "元素-2")), make_chunk("id/ two"))
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(to_row(item), ensure_ascii=False) for item in chunks), encoding="utf-8")
    directory = tmp_path / "index"

    pipeline = make_pipeline()
    pipeline.build_and_persist(artifact, directory)
    result = pipeline.load(directory).search("query", top_k=1)[0]

    assert result.chunk.chunk_id == "id:one"
    assert result.chunk.element_ids == ("element/1", "元素-2")
    assert result.chunk.source_file == "source ü.pdf"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.pop("text"),
        lambda row: row.update(text=""),
        lambda row: row.update(page_start="2"),
        lambda row: row.update(element_ids=["", 3]),
    ],
)
def test_corpus_loader_rejects_partial_or_malformed_records(tmp_path, mutator) -> None:
    row = to_row(make_chunk("one"))
    mutator(row)
    artifact = tmp_path / "bad.jsonl"
    artifact.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(CorpusChunkLoadError):
        CorpusChunkLoader().load(artifact)


def test_persistence_rejects_changed_chunk_payload(tmp_path) -> None:
    chunks = (make_chunk("one"), make_chunk("two"))
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(to_row(item)) for item in chunks), encoding="utf-8")
    directory = tmp_path / "index"
    pipeline = make_pipeline()
    pipeline.build_and_persist(artifact, directory)

    persisted = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    persisted[0]["text"] = "tampered payload"
    (directory / "chunks.json").write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="fingerprint"):
        pipeline.load(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("embedding_dimension", 99, "dimension"), ("normalize_embeddings", False, "normalization")],
)
def test_persistence_rejects_incompatible_metadata(tmp_path, field, value, message) -> None:
    chunks = (make_chunk("one"), make_chunk("two"))
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(to_row(item)) for item in chunks), encoding="utf-8")
    directory = tmp_path / "index"
    pipeline = make_pipeline()
    pipeline.build_and_persist(artifact, directory)

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata[field] = value
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match=message):
        pipeline.load(directory)