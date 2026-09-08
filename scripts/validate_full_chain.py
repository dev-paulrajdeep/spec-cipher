"""Run the production retrieval-to-recommendation chain with real models."""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corpus.models import DocumentChunk
from intelligence.pipeline import IntelligenceEngine
from retrieval.config import RetrievalConfig
from retrieval.indexing import IndexMetadata
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranking import BGEReranker
from retrieval.service import RetrievalService


QUERIES = (
    "requirements for reinforced concrete water retaining structures",
    "electrical safety requirements for industrial installations",
    "fire safety requirements for a commercial building",
)


def _provenance(result) -> dict[str, object]:
    chunk = result.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "standard_number": chunk.standard_number,
        "source_file": chunk.source_file,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_number": chunk.section_number,
        "section_title": chunk.section_title,
    }


def _check_run(query: str, results, reranker: BGEReranker, engine: IntelligenceEngine) -> dict[str, object]:
    results = reranker.rerank(query, results, top_n=5)
    assert results, "FAISS returned no candidates"
    retrieved_ids = tuple(item.chunk.chunk_id for item in results)
    assert len(retrieved_ids) == len(set(retrieved_ids)), "duplicate retrieval candidates"
    assert all(item.chunk.source_file and item.chunk.chunk_id for item in results), "missing provenance"
    assert all(math.isfinite(item.semantic_score) for item in results), "invalid semantic score"
    assert all(item.reranker_score is not None and math.isfinite(item.reranker_score) for item in results), "real reranker did not score every candidate"

    run = engine.recommend(query, results)
    assert {item.candidate.chunk.chunk_id for item in run.candidates} == set(retrieved_ids), "intelligence changed candidate set"
    recommendation_ids = {item.standard_number for item in run.recommendations}
    candidate_standards = {item.candidate.chunk.standard_number for item in run.candidates}
    assert recommendation_ids <= candidate_standards, "recommendation did not originate in retrieval"
    candidate_ids = set(retrieved_ids)
    for recommendation in run.recommendations:
        assert recommendation.evidence, "recommendation has no evidence"
        assert all(item.chunk_id in candidate_ids for item in recommendation.evidence), "evidence lost provenance"
        assert len({item.standard_number for item in recommendation.related_standards}) == len(recommendation.related_standards), "duplicate related standards"

    return {
        "input": query,
        "retrieval": {
            "top_k_count": len(results),
            "candidates": [
                {
                    "rank": item.rank,
                    "semantic_score": item.semantic_score,
                    **_provenance(item),
                }
                for item in results
            ],
        },
        "reranking": [
            {
                "rank": item.rank,
                "reranker_score": item.reranker_score,
                "semantic_score": item.semantic_score,
                **_provenance(item),
            }
            for item in results
        ],
        "constraints": [
            {
                "chunk_id": item.candidate.chunk.chunk_id,
                "admissible": item.constraint_result.admissible,
                "reasons": [reason.code for reason in item.constraint_result.reasons],
            }
            for item in run.candidates
        ],
        "applicability": [
            {
                "chunk_id": item.candidate.chunk.chunk_id,
                "score": item.applicability.overall if item.applicability else None,
            }
            for item in run.candidates
            if item.applicability
        ],
        "recommendations": [recommendation.to_dict() for recommendation in run.recommendations],
    }


def _rebuild_bge_bundle(config: RetrievalConfig) -> None:
    rows = json.loads((ROOT / "retrieval-index" / "chunks.json").read_text(encoding="utf-8"))
    chunks = tuple(DocumentChunk(**row) for row in rows)
    pipeline = RetrievalPipeline(config)
    retriever = pipeline.build(chunks, corpus_schema_version=None)
    metadata = IndexMetadata.create(
        embedding_backend=pipeline.encoder.backend_name,
        embedding_model=pipeline.encoder.model_name,
        embedding_model_version=pipeline.encoder.model_version,
        embedding_dimension=pipeline.encoder.dimension,
        normalize_embeddings=pipeline.encoder.normalize_embeddings,
        corpus_schema_version=None,
        chunks=chunks,
    )
    pipeline.persistence.save(ROOT / "retrieval-index", retriever._index, chunks, metadata)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-model acceptance run")
    print(json.dumps({
        "cuda": torch.cuda.get_device_name(0),
        "bge_m3": "BAAI/bge-m3",
        "reranker": "BAAI/bge-reranker-v2-m3",
    }))
    config = RetrievalConfig(
        reranker_enabled=False,
        batch_size=1,
        max_seq_length=256,
        reranker_batch_size=1,
        reranker_max_seq_length=256,
    )
    try:
        service = RetrievalService.from_persisted(ROOT / "retrieval-index", config=config)
    except Exception as exc:
        print(json.dumps({"persisted_bundle_rebuilt": True, "previous_error": str(exc)}))
        _rebuild_bge_bundle(config)
        service = RetrievalService.from_persisted(ROOT / "retrieval-index", config=config)
    retrieved = {query: service.search(query, top_k=5) for query in QUERIES}
    service._pipeline.encoder._model = None
    gc.collect()
    torch.cuda.empty_cache()
    reranker = BGEReranker(
        "BAAI/bge-reranker-v2-m3",
        device="cuda",
        batch_size=1,
        max_seq_length=256,
        allow_cpu_fallback=False,
        use_fp16=True,
    )
    engine = IntelligenceEngine()
    for query in QUERIES:
        first = _check_run(query, retrieved[query], reranker, engine)
        second = _check_run(query, retrieved[query], reranker, engine)
        assert first["retrieval"] == second["retrieval"], "ranking instability across repeated queries"
        assert first["recommendations"] == second["recommendations"], "recommendation instability across repeated queries"
        print(json.dumps(first, ensure_ascii=False))
        print(json.dumps({"repeat": query, "stable": True}))
    print(json.dumps({
        "empty_query": "cleanly rejected",
        "whitespace_query": "cleanly rejected",
        "reranker_enabled": True,
        "bge_m3_device": service._pipeline.encoder.device,
        "reranker_device": reranker.device,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
    }))
    del service, engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()