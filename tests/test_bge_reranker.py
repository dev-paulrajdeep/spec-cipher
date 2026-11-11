"""Optional real BGE-M3 plus BGE-reranker-v2-m3 GPU smoke test."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from corpus.models import DocumentChunk
from retrieval.config import RetrievalConfig
from retrieval.indexing import IndexMetadata
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranking import BGEReranker

QUERY = "requirements for reinforced concrete water retaining structures"


CHUNKS = (
    DocumentChunk(
        "smoke-concrete-water", "IS 3370 (Part 1):2021", "concrete-water-retaining.pdf",
        12, 13, "4", "General Requirements",
        "Requirements for reinforced concrete water retaining structures, liquid tightness, durability, crack control, and hydrostatic pressure.",
        ("clause-4.1", "table-4A"),
    ),
    DocumentChunk(
        "smoke-structural-steel", "IS 800:2007", "structural-steel.pdf",
        21, 22, "5", "Design of Steel Structures",
        "Structural steel design requirements for buildings, bolted connections, welded joints, member stability, and steel load resistance.",
        ("clause-5.2",),
    ),
    DocumentChunk(
        "smoke-electrical-installations", "IS 732:2019", "electrical-installations.pdf",
        7, 8, "3", "Electrical Installations",
        "Electrical installation requirements for buildings, wiring systems, protective devices, earthing, and electrical safety.",
        ("clause-3.1",),
    ),
    DocumentChunk(
        "smoke-concrete-tanks", "IS 456:2000", "plain-reinforced-concrete.pdf",
        30, 31, "10", "Reinforced Concrete",
        "Plain and reinforced concrete construction provisions covering materials, structural design, serviceability, and reinforcement detailing.",
        ("clause-10.3",),
    ),
)


def memory(label: str) -> None:
    print(f"{label}_allocated_gb: {torch.cuda.memory_allocated() / 1024**3:.4f}")
    print(f"{label}_reserved_gb: {torch.cuda.memory_reserved() / 1024**3:.4f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this optional smoke test")
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    memory("before_loading")

    config = RetrievalConfig(
        embedding_model="BAAI/bge-m3",
        embedding_model_version="main",
        embedding_dimension=1024,
        embedding_use_fp16=True,
        embedding_device="cuda",
        allow_cpu_fallback=False,
        batch_size=2,
        top_k=4,
    )
    pipeline = RetrievalPipeline(config)
    retriever = pipeline.build(CHUNKS, corpus_schema_version="smoke-1.0")
    print("bge_m3_model:", pipeline.encoder.model_name)
    print("bge_m3_device:", pipeline.encoder.device)
    print("bge_m3_dimension:", pipeline.encoder.dimension)
    print("bge_m3_norms:", [float(norm) for norm in np.linalg.norm(
        pipeline.encoder.encode_batch([chunk.text for chunk in CHUNKS]), axis=1
    )])
    memory("after_bge_m3")

    reranker_candidates = retriever.search(QUERY, top_k=4)
    with TemporaryDirectory(prefix="spec-cipher-bge-reranker-smoke-") as temp:
        directory = Path(temp) / "index"
        metadata = IndexMetadata.create(
            embedding_backend=pipeline.encoder.backend_name,
            embedding_model=pipeline.encoder.model_name,
            embedding_model_version=pipeline.encoder.model_version,
            embedding_dimension=pipeline.encoder.dimension,
            normalize_embeddings=pipeline.encoder.normalize_embeddings,
            corpus_schema_version="smoke-1.0",
            chunks=CHUNKS,
        )
        pipeline.persistence.save(directory, retriever._index, CHUNKS, metadata)
        reloaded = pipeline.load(directory)
        semantic_candidates = reloaded.search(QUERY, top_k=4)
        print("persistence_reload: true")
        print("semantic_candidates_equal_before_reload:", semantic_candidates == reranker_candidates)

        pipeline.encoder._model = None
        del pipeline
        del retriever
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        memory("after_bge_m3_release")

        reranker = BGEReranker(
            "BAAI/bge-reranker-v2-m3",
            model_version="main",
            device="cuda",
            batch_size=1,
            max_seq_length=256,
            allow_cpu_fallback=False,
            use_fp16=True,
        )
        print("reranker_model:", reranker.model_name)
        print("reranker_device_before_load:", reranker.device)
        reranked = reranker.rerank(QUERY, semantic_candidates, top_n=4)
        print("candidate_count:", len(reranked))
        print("persisted_reload: true")
        print("reranker_device_after_load:", reranker.device)
        memory("after_reranker")
        print("ranked_results:")
        for result in reranked:
            chunk = result.chunk
            print(json.dumps({
                "rank": result.rank,
                "semantic_score": result.semantic_score,
                "reranker_score": result.reranker_score,
                "chunk_id": chunk.chunk_id,
                "standard_number": chunk.standard_number,
                "source_file": chunk.source_file,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_number": chunk.section_number,
                "section_title": chunk.section_title,
                "element_ids": list(chunk.element_ids),
            }, ensure_ascii=False))
    print("peak_allocated_gb:", torch.cuda.max_memory_allocated() / 1024**3)
    print("peak_reserved_gb:", torch.cuda.max_memory_reserved() / 1024**3)
    print("oom: false")


if __name__ == "__main__":
    main()
