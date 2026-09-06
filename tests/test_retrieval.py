from __future__ import annotations

import json

import numpy as np
import pytest

from corpus.models import DocumentChunk
from retrieval.config import RetrievalConfig
from retrieval.corpus_loader import CorpusChunkLoadError, CorpusChunkLoader
from retrieval.evaluation import EvaluationCase, evaluate, mrr_at_k, precision_at_k, recall_at_k
from retrieval.indexing import FaissIndex, IndexCompatibilityError
from retrieval.models import RetrievalResult, group_results_by_standard
from retrieval.pipeline import RetrievalPipeline
from retrieval.service import RetrievalService, RetrievalUnavailableError


class StaticEncoder:
    """Deterministic unit vectors for offline FAISS and pipeline tests."""

    def __init__(self, vectors: dict[str, list[float]], model_name: str = "test-encoder") -> None:
        self._vectors = {key: np.asarray(value, dtype=np.float32) for key, value in vectors.items()}
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return len(next(iter(self._vectors.values())))

    @property
    def normalize_embeddings(self) -> bool:
        return True

    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("empty")
        return self._vectors[text].copy()

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.encode_text(text) for text in texts]).astype(np.float32)


def chunk(identifier: str, standard: str, text: str, page: int = 1) -> DocumentChunk:
    return DocumentChunk(identifier, standard, f"{standard}.pdf", page, page, "1", "Scope", text)


@pytest.fixture
def chunks() -> tuple[DocumentChunk, ...]:
    return (
        chunk("c-waterproof", "IS 2645:2003", "waterproof concrete compound", 2),
        chunk("c-storage", "IS 3370 (Part 1):2021", "concrete liquid storage structure", 3),
        chunk("c-test", "IS 2645:2003", "compound test method", 4),
    )


@pytest.fixture
def encoder(chunks) -> StaticEncoder:
    return StaticEncoder(
        {
            chunks[0].text: [1.0, 0.0, 0.0],
            chunks[1].text: [0.0, 1.0, 0.0],
            chunks[2].text: [0.8, 0.6, 0.0],
            "waterproofing compound for concrete": [1.0, 0.0, 0.0],
            "liquid storage": [0.0, 1.0, 0.0],
        }
    )


def pipeline(encoder: StaticEncoder) -> RetrievalPipeline:
    return RetrievalPipeline(RetrievalConfig(embedding_model="test-encoder", top_k=2), encoder=encoder)


def chunk_row(item: DocumentChunk) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "standard_number": item.standard_number,
        "source_file": item.source_file,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "section_number": item.section_number,
        "section_title": item.section_title,
        "text": item.text,
    }


def test_loader_reads_canonical_json_and_jsonl(tmp_path, chunks) -> None:
    canonical = tmp_path / "standard.json"
    canonical.write_text(json.dumps({"schema_version": "1.0", "chunks": [chunk_row(item) for item in chunks]}), encoding="utf-8")
    loaded = CorpusChunkLoader().load(canonical)
    assert loaded.schema_version == "1.0"
    assert loaded.chunks == chunks
    jsonl = tmp_path / "chunks.jsonl"
    jsonl.write_text("\n".join(json.dumps(chunk_row(item)) for item in chunks), encoding="utf-8")
    assert CorpusChunkLoader().load(jsonl).chunks == chunks


def test_loader_rejects_malformed_and_duplicate_chunks(tmp_path, chunks) -> None:
    malformed = tmp_path / "bad.jsonl"
    row = chunk_row(chunks[0])
    del row["text"]
    malformed.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(CorpusChunkLoadError, match="text"):
        CorpusChunkLoader().load(malformed)
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text("\n".join([json.dumps(chunk_row(chunks[0]))] * 2), encoding="utf-8")
    with pytest.raises(CorpusChunkLoadError, match="duplicate"):
        CorpusChunkLoader().load(duplicate)


def test_loader_preserves_unknown_standard_identifier(tmp_path, chunks) -> None:
    unknown_standard = chunk_row(chunks[0])
    unknown_standard["standard_number"] = None
    artifact = tmp_path / "special-publication.jsonl"
    artifact.write_text(json.dumps(unknown_standard), encoding="utf-8")
    loaded = CorpusChunkLoader().load(artifact)
    assert loaded.chunks[0].standard_number is None


def test_loader_preserves_optional_element_provenance(tmp_path, chunks) -> None:
    row = chunk_row(chunks[0])
    row["element_ids"] = ["text-block-17", "table-3"]
    artifact = tmp_path / "with-elements.jsonl"
    artifact.write_text(json.dumps(row), encoding="utf-8")

    loaded = CorpusChunkLoader().load(artifact)

    assert loaded.chunks[0].element_ids == ("text-block-17", "table-3")


def test_loader_rejects_invalid_element_provenance(tmp_path, chunks) -> None:
    row = chunk_row(chunks[0])
    row["element_ids"] = ["", 7]
    artifact = tmp_path / "bad-elements.jsonl"
    artifact.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(CorpusChunkLoadError, match="element_ids"):
        CorpusChunkLoader().load(artifact)


def test_embedding_batch_is_repeatable_and_has_expected_dimensions(encoder, chunks) -> None:
    first = encoder.encode_batch([item.text for item in chunks])
    second = encoder.encode_batch([item.text for item in chunks])
    assert first.shape == (3, 3)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_faiss_index_maps_ids_and_caps_top_k() -> None:
    index = FaissIndex(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ["first", "second"])
    results = index.search(np.asarray([[1.0, 0.0]], dtype=np.float32), top_k=5)[0]
    assert [result.chunk_id for result in results] == ["first", "second"]
    assert results[0].similarity_score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unit-normalized"):
        index.add(np.asarray([[2.0, 0.0]], dtype=np.float32), ["third"])


def test_semantic_top_k_preserves_chunk_provenance_and_groups_candidates(encoder, chunks) -> None:
    retriever = pipeline(encoder).build(chunks, corpus_schema_version="1.0")
    results = retriever.search("waterproofing compound for concrete", top_k=5)
    assert [item.chunk.chunk_id for item in results] == ["c-waterproof", "c-test", "c-storage"]
    assert results[0].rank == 1
    assert results[0].chunk.source_file == "IS 2645:2003.pdf"
    candidates = group_results_by_standard(results)
    assert candidates[0].standard_number == "IS 2645:2003"
    assert len(candidates[0].results) == 2
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("")
    for invalid_query in (None, 42):
        with pytest.raises(ValueError, match="non-empty"):
            retriever.search(invalid_query)  # type: ignore[arg-type]
    for invalid_top_k in (0, -1, "2", True):
        with pytest.raises(ValueError, match="top_k"):
            retriever.search("waterproofing compound for concrete", invalid_top_k)  # type: ignore[arg-type]


def test_public_service_contract_hides_index_and_serializes_provenance(tmp_path, encoder, chunks) -> None:
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(chunk_row(item)) for item in chunks), encoding="utf-8")
    pipeline(encoder).build_and_persist(artifact, tmp_path / "index")

    service = RetrievalService(_pipeline=pipeline(encoder))
    with pytest.raises(RetrievalUnavailableError, match="not loaded"):
        service.search("waterproofing compound for concrete")
    service.load(tmp_path / "index")

    results = service.search("waterproofing compound for concrete", top_k=2)
    assert len(results) == 2
    assert not hasattr(service, "index")
    response = results[0].to_dict()
    assert response == {
        "standard_number": "IS 2645:2003",
        "title": None,
        "section_number": "1",
        "section_title": "Scope",
        "text": "waterproof concrete compound",
        "score": pytest.approx(1.0),
        "rank": 1,
        "source": {
            "chunk_id": "c-waterproof",
            "source_file": "IS 2645:2003.pdf",
            "page_start": 2,
            "page_end": 2,
            "element_ids": [],
        },
    }


def test_duplicate_chunks_and_empty_corpus_cannot_be_indexed(encoder, chunks) -> None:
    retrieval = pipeline(encoder)
    with pytest.raises(ValueError, match="empty corpus"):
        retrieval.build(())
    with pytest.raises(ValueError, match="duplicate"):
        retrieval.build((chunks[0], chunks[0]))


def test_index_persistence_reload_and_configuration_compatibility(tmp_path, encoder, chunks) -> None:
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(chunk_row(item)) for item in chunks), encoding="utf-8")
    saved = pipeline(encoder).build_and_persist(artifact, tmp_path / "index")
    loaded = pipeline(encoder).load(tmp_path / "index")
    assert loaded.search("liquid storage")[0].chunk.chunk_id == "c-storage"
    assert saved.search("liquid storage")[0].chunk.chunk_id == "c-storage"
    assert loaded.search("liquid storage") == saved.search("liquid storage")
    incompatible = RetrievalPipeline(
        RetrievalConfig(embedding_model="other-encoder"),
        encoder=StaticEncoder(encoder._vectors, model_name="other-encoder"),
    )
    with pytest.raises(IndexCompatibilityError, match="different embedding model"):
        incompatible.load(tmp_path / "index")


def test_load_rejects_missing_or_corrupted_persistence_artifacts(tmp_path, encoder, chunks) -> None:
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(chunk_row(item)) for item in chunks), encoding="utf-8")
    index_directory = tmp_path / "index"
    retrieval = pipeline(encoder)
    retrieval.build_and_persist(artifact, index_directory)

    with pytest.raises(FileNotFoundError, match="incomplete"):
        retrieval.load(tmp_path / "missing-index")

    (index_directory / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IndexCompatibilityError, match="metadata"):
        retrieval.load(index_directory)

    retrieval.build_and_persist(artifact, index_directory)
    (index_directory / "chunk_ids.json").write_text(json.dumps(["wrong-id"] * len(chunks)), encoding="utf-8")
    with pytest.raises(IndexCompatibilityError, match="mapping"):
        retrieval.load(index_directory)

    retrieval.build_and_persist(artifact, index_directory)
    faiss = FaissIndex._import_faiss()
    wrong_metric_index = faiss.IndexFlatL2(encoder.dimension)
    wrong_metric_index.add(encoder.encode_batch([item.text for item in chunks]))
    faiss.write_index(wrong_metric_index, str(index_directory / "index.faiss"))
    with pytest.raises(IndexCompatibilityError, match="IndexFlatIP"):
        retrieval.load(index_directory)


def test_end_to_end_corpus_jsonl_embeddings_faiss_query_top_k(tmp_path, encoder, chunks) -> None:
    artifact = tmp_path / "chunks.jsonl"
    artifact.write_text("\n".join(json.dumps(chunk_row(item)) for item in chunks), encoding="utf-8")
    results = pipeline(encoder).build_from_corpus(artifact).search("waterproofing compound for concrete")
    assert results[0].chunk.standard_number == "IS 2645:2003"
    assert results[0].similarity_score == pytest.approx(1.0)


def test_evaluation_metrics_and_gold_runner(encoder, chunks) -> None:
    retriever = pipeline(encoder).build(chunks)
    case = EvaluationCase("waterproofing compound for concrete", frozenset({"IS 2645:2003"}))
    results = retriever.search(case.query, top_k=2)
    assert recall_at_k(results, case, 1) == 1.0
    assert precision_at_k(results, case, 2) == 1.0
    assert mrr_at_k(results, case, 2) == 1.0
    summary = evaluate(retriever, [case, EvaluationCase("liquid storage", frozenset({"IS 3370 (Part 1):2021"}))], k=1)
    assert summary.recall_at_k == summary.precision_at_k == summary.mrr_at_k == 1.0
