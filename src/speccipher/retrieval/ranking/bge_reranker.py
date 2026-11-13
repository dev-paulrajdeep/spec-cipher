"""Lazy cross-encoder and deterministic fallback for fused candidates."""

from __future__ import annotations

from math import exp, isfinite

from speccipher.domain.candidates import Candidate, RerankScore


class BGERerankerAdapter:
    """Adapter to the existing lazy BGE v2 M3 cross-encoder implementation.

    The Sentence Transformers, PyTorch, and NumPy imports stay inside ``score``;
    application construction and module import do not load model weights.
    """

    strategy = "bge_reranker_v2_m3"

    def __init__(
        self,
        model_name: str,
        *,
        model_version: str = "main",
        device: str = "auto",
        batch_size: int = 16,
        max_seq_length: int | None = None,
        allow_cpu_fallback: bool = False,
        use_fp16: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self._options = {
            "model_name": model_name,
            "model_version": model_version,
            "device": device,
            "batch_size": batch_size,
            "max_seq_length": max_seq_length,
            "allow_cpu_fallback": allow_cpu_fallback,
            "use_fp16": use_fp16,
            "local_files_only": local_files_only,
        }
        self._adapter = None

    def _get_adapter(self):
        if self._adapter is None:
            from retrieval.reranking.bge import BGEReranker

            self._adapter = BGEReranker(**self._options)
        return self._adapter

    def score(self, requirement: str, candidates: tuple[Candidate, ...]) -> dict[str, RerankScore]:
        if not candidates:
            return {}
        from corpus.models import DocumentChunk as LegacyDocumentChunk
        from retrieval.models import RetrievalResult

        legacy_results = tuple(
            RetrievalResult(
                chunk=LegacyDocumentChunk(
                    chunk_id=candidate.chunk.chunk_id,
                    standard_number=(
                        candidate.chunk.standard_id.code if candidate.chunk.standard_id else None
                    ),
                    source_file=candidate.chunk.source_file_id,
                    page_start=candidate.chunk.absolute_page_start,
                    page_end=candidate.chunk.absolute_page_end,
                    section_number=candidate.chunk.section_number,
                    section_title=candidate.chunk.section_title,
                    text=candidate.chunk.text,
                ),
                similarity_score=candidate.dense_score or 0.0,
                rank=rank,
            )
            for rank, candidate in enumerate(candidates, start=1)
        )
        ranked = self._get_adapter().rerank(requirement, legacy_results)
        raw = {item.chunk.chunk_id: float(item.reranker_score) for item in ranked}
        if set(raw) != {candidate.chunk.chunk_id for candidate in candidates}:
            raise RuntimeError("Cross-encoder did not return one score per candidate")
        if any(not isfinite(value) for value in raw.values()):
            raise RuntimeError("Cross-encoder returned a non-finite score")
        return {
            chunk_id: RerankScore(raw_score=value, normalized_score=_sigmoid(value))
            for chunk_id, value in raw.items()
        }

    def close(self) -> None:
        """Release cross-encoder weights and device memory if they were loaded."""
        if self._adapter is not None:
            self._adapter.release()
            self._adapter = None


def _sigmoid(value: float) -> float:
    """Bound a logit for score fusion without min-maxing the current candidate set."""
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)
