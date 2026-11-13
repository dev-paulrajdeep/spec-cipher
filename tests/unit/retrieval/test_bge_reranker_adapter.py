from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from retrieval.reranking.bge import BGEReranker

from speccipher.domain.candidates import Candidate
from speccipher.domain.identity import StandardId
from speccipher.domain.models import DocumentChunk
from speccipher.retrieval.ranking.bge_reranker import BGERerankerAdapter


def _candidate(chunk_id: str, text: str) -> Candidate:
    return Candidate(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            standard_id=StandardId.parse("IS 456:2000"),
            logical_doc_id="pub_is456_2000",
            source_file_id="is456.pdf",
            absolute_page_start=1,
            absolute_page_end=1,
            text=text,
        ),
        dense_score=0.8,
        rrf_score=0.03,
    )


def test_bge_adapter_loads_project_adapter_lazily_and_preserves_raw_logits(
    monkeypatch,
) -> None:
    instances = []

    class FakeBGEReranker:
        def __init__(self, **options):
            self.options = options
            self.closed = False
            instances.append(self)

        def rerank(self, query, candidates):
            return tuple(
                SimpleNamespace(
                    chunk=SimpleNamespace(chunk_id=candidate.chunk.chunk_id),
                    reranker_score=raw_score,
                )
                for candidate, raw_score in zip(candidates, (-2.0, 2.0), strict=True)
            )

        def release(self):
            self.closed = True

    module = ModuleType("retrieval.reranking.bge")
    module.BGEReranker = FakeBGEReranker
    monkeypatch.setitem(sys.modules, "retrieval.reranking.bge", module)

    adapter = BGERerankerAdapter("BAAI/bge-reranker-v2-m3", device="cpu")
    assert adapter._adapter is None

    scores = adapter.score(
        "concrete structures",
        (_candidate("one", "concrete"), _candidate("two", "concrete structures")),
    )

    assert instances[0].options["device"] == "cpu"
    assert scores["one"].raw_score == -2.0
    assert scores["two"].raw_score == 2.0
    assert scores["one"].normalized_score < 0.5
    assert scores["two"].normalized_score > 0.5
    adapter.close()
    assert instances[0].closed is True


def test_cross_encoder_uses_pinned_revision_and_local_cache_only(monkeypatch) -> None:
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, **options):
            captured["model_name"] = model_name
            captured.update(options)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float16 = object()
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    model = BGEReranker(
        "BAAI/bge-reranker-v2-m3",
        model_version="pinned-revision",
        device="cpu",
        local_files_only=True,
    )._load_model()

    assert isinstance(model, FakeCrossEncoder)
    assert captured["model_name"] == "BAAI/bge-reranker-v2-m3"
    assert captured["revision"] == "pinned-revision"
    assert captured["local_files_only"] is True
    assert captured["device"] == "cpu"
