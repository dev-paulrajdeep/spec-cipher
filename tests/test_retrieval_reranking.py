from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from corpus.models import DocumentChunk
from retrieval.config import RetrievalConfig
from retrieval.models import RetrievalResult
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranking import BGEReranker


class StaticEncoder:
    backend_name = "test-encoder"
    model_name = "test-encoder"
    model_version = "test"
    dimension = 3
    normalize_embeddings = True

    def encode_text(self, text: str) -> np.ndarray:
        return np.asarray({
            "water query": [1.0, 0.0, 0.0],
            "water concrete": [1.0, 0.0, 0.0],
            "steel structure": [0.0, 1.0, 0.0],
            "electrical installation": [0.0, 0.0, 1.0],
        }[text], dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.encode_text(text) for text in texts])


def result(identifier: str, text: str, rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        DocumentChunk(identifier, f"IS {identifier}", f"{identifier}.pdf", 2, 3, "1", "Scope", text, (f"element-{identifier}",)),
        score,
        rank,
    )


class FakeReranker:
    backend_name = "fake-reranker"
    model_name = "fake-model"
    model_version = "test"

    def rerank(self, query, candidates, top_n=None):
        ordered = sorted(candidates, key=lambda candidate: candidate.chunk.chunk_id, reverse=True)
        return tuple(ordered[:top_n] if top_n is not None else ordered)


def test_config_validates_optional_reranker_settings() -> None:
    config = RetrievalConfig(reranker_enabled=True, reranker_candidate_k=8, reranker_max_seq_length=256)
    assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert config.reranker_candidate_k == 8
    with pytest.raises(ValueError, match="reranker_device"):
        RetrievalConfig(reranker_device="tpu")
    with pytest.raises(ValueError, match="reranker_batch_size"):
        RetrievalConfig(reranker_batch_size=0)
    with pytest.raises(ValueError, match="reranker_candidate_k"):
        RetrievalConfig(reranker_candidate_k=0)
    with pytest.raises(ValueError, match="reranker_use_fp16"):
        RetrievalConfig(reranker_use_fp16=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding_use_fp16"):
        RetrievalConfig(embedding_use_fp16=1)  # type: ignore[arg-type]


def test_bge_reranker_is_lazy_and_handles_empty_candidates() -> None:
    reranker = BGEReranker(device="cpu")
    assert reranker._model is None
    assert reranker.rerank("query", ()) == ()
    assert reranker._model is None


def test_bge_reranker_batches_pairs_and_preserves_scores_and_provenance() -> None:
    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, pairs, **kwargs):
            self.calls.append((pairs, kwargs))
            return [0.2, 0.9, 0.4]

    reranker = BGEReranker(device="cpu", batch_size=2)
    model = FakeModel()
    reranker._model = model
    candidates = (
        result("one", "water text", 1, 0.91),
        result("two", "steel text", 2, 0.82),
        result("three", "electrical text", 3, 0.73),
    )

    ranked = reranker.rerank("water query", candidates, top_n=2)

    assert [item.chunk.chunk_id for item in ranked] == ["two", "three"]
    assert [item.rank for item in ranked] == [1, 2]
    assert [item.semantic_score for item in ranked] == [0.82, 0.73]
    assert [item.reranker_score for item in ranked] == [0.9, 0.4]
    assert ranked[0].chunk.element_ids == ("element-two",)
    pairs, kwargs = model.calls[0]
    assert pairs == [("water query", "water text"), ("water query", "steel text"), ("water query", "electrical text")]
    assert kwargs["batch_size"] == 2


def test_bge_reranker_ties_use_original_rank_then_chunk_id() -> None:
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [0.5] * len(pairs)

    reranker = BGEReranker(device="cpu")
    reranker._model = FakeModel()
    candidates = (
        result("z", "z", 2, 0.2),
        result("a", "a", 1, 0.8),
    )
    ranked = reranker.rerank("query", candidates)
    assert [item.chunk.chunk_id for item in ranked] == ["a", "z"]
    assert [item.reranker_score for item in ranked] == [0.5, 0.5]


def test_bge_reranker_rejects_invalid_scores_and_queries() -> None:
    class FakeModel:
        def predict(self, pairs, **kwargs):
            return [float("nan")] * len(pairs)

    reranker = BGEReranker(device="cpu")
    reranker._model = FakeModel()
    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank("", (result("one", "text", 1, 0.2),))
    with pytest.raises(RuntimeError, match="invalid score"):
        reranker.rerank("query", (result("one", "text", 1, 0.2),))


def test_bge_reranker_cpu_policy_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    sentence_transformers = ModuleType("sentence_transformers")
    calls = []

    def cross_encoder(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace()

    sentence_transformers.CrossEncoder = cross_encoder
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    with pytest.raises(RuntimeError, match="allow_cpu_fallback"):
        BGEReranker(device="cuda")._load_model()
    fallback = BGEReranker(
        device="cuda",
        allow_cpu_fallback=True,
        model_version="test-revision",
        local_files_only=True,
    )
    fallback._load_model()
    assert fallback.device == "cpu"
    assert calls[0][1]["revision"] == "test-revision"
    assert calls[0][1]["local_files_only"] is True


def test_pipeline_retrieval_reranker_runs_after_faiss_and_preserves_identity() -> None:
    chunks = (
        DocumentChunk("water", "IS water", "water.pdf", 1, 1, "1", "Scope", "water concrete"),
        DocumentChunk("steel", "IS steel", "steel.pdf", 2, 2, "1", "Scope", "steel structure"),
        DocumentChunk("electrical", "IS electrical", "electrical.pdf", 3, 3, "1", "Scope", "electrical installation"),
    )
    pipeline = RetrievalPipeline(
        RetrievalConfig(embedding_model="test-encoder", top_k=2, reranker_candidate_k=3),
        encoder=StaticEncoder(),
        reranker=FakeReranker(),
    )
    ranked = pipeline.build(chunks).search("water query", top_k=2)
    assert [item.chunk.chunk_id for item in ranked] == ["water", "steel"]
    assert [item.semantic_score for item in ranked] == [1.0, 0.0]
    assert all(item.reranker_score is None for item in ranked)


def test_reranked_result_json_exposes_both_scores() -> None:
    item = RetrievalResult(result("one", "water text", 1, 0.8).chunk, 0.8, 1, 0.95)
    payload = item.to_dict()
    assert payload["semantic_score"] == 0.8
    assert payload["reranker_score"] == 0.95
    assert payload["score"] == 0.8
