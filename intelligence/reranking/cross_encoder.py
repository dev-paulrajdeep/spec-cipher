"""Optional lazy Sentence Transformers CrossEncoder reranker adapter."""

from __future__ import annotations

import numpy as np

from intelligence.models import RerankResult
from retrieval.models import RetrievalResult
from .reranker import RerankedCandidate


class CrossEncoderReranker:
    """Dedicated model adapter; lazy loading keeps LLM/model dependencies optional."""

    strategy = "cross_encoder"

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def rerank(self, requirement: str, candidates: tuple[RetrievalResult, ...]) -> tuple[RerankedCandidate, ...]:
        if not candidates:
            return ()
        scores = np.asarray(
            self._load_model().predict([(requirement, item.chunk.text) for item in candidates], batch_size=self._batch_size),
            dtype=np.float64,
        ).reshape(-1)
        normalised = self._normalise(scores)
        rows = sorted(
            ((float(raw), float(score), original_rank, candidate) for original_rank, (raw, score, candidate) in enumerate(zip(scores, normalised, candidates, strict=True), start=1)),
            key=lambda item: (-item[1], item[2], item[3].chunk.chunk_id),
        )
        return tuple(
            RerankedCandidate(candidate, RerankResult(raw, score, rank, self.strategy))
            for rank, (raw, score, _, candidate) in enumerate(rows, start=1)
        )

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 1 or np.isclose(scores.max(), scores.min()):
            return np.ones_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc
            self._model = CrossEncoder(self._model_name)
        return self._model
