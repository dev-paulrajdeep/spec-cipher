"""Lazy BGE-reranker-v2-m3 adapter for FAISS-returned candidates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from retrieval.models import RetrievalResult


class BGEReranker:
    """Use a cross-encoder only on the supplied first-stage candidates."""

    backend_name = "sentence-transformers-cross-encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        model_version: str = "main",
        device: str = "auto",
        batch_size: int = 16,
        max_seq_length: int | None = None,
        allow_cpu_fallback: bool = False,
        use_fp16: bool = True,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_seq_length is not None and max_seq_length < 1:
            raise ValueError("max_seq_length must be positive when provided")
        if not isinstance(use_fp16, bool):
            raise ValueError("use_fp16 must be a boolean")
        self._model_name = model_name
        self._model_version = model_version
        self._device = device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._allow_cpu_fallback = allow_cpu_fallback
        self._use_fp16 = use_fp16
        self._model = None
        self._resolved_device: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        return self._resolved_device or self._device

    def rerank(
        self,
        query: str,
        candidates: tuple[RetrievalResult, ...],
        top_n: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_n is not None and (not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1):
            raise ValueError("top_n must be positive")
        if not candidates:
            return ()
        raw_scores = np.asarray(
            self._load_model().predict(
                [(query, candidate.chunk.text) for candidate in candidates],
                batch_size=self._batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if len(raw_scores) != len(candidates) or not np.isfinite(raw_scores).all():
            raise RuntimeError("Reranker returned an invalid score count or non-finite score")
        ranked = sorted(
            zip(raw_scores.tolist(), candidates, strict=True),
            key=lambda item: (-item[0], item[1].rank, item[1].chunk.chunk_id),
        )
        limit = len(ranked) if top_n is None else min(top_n, len(ranked))
        return tuple(
            replace(candidate, rank=rank, reranker_score=float(score))
            for rank, (score, candidate) in enumerate(ranked[:limit], start=1)
        )

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch and sentence-transformers are required for BGE reranking"
                ) from exc
            if self._device == "cpu":
                self._resolved_device = "cpu"
            elif torch.cuda.is_available():
                self._resolved_device = "cuda"
            elif self._allow_cpu_fallback:
                self._resolved_device = "cpu"
            else:
                raise RuntimeError(
                    "CUDA is unavailable for the configured reranker device; "
                    "set allow_cpu_fallback=True or device='cpu' explicitly"
                )
            model_kwargs = {"torch_dtype": torch.float16} if self._use_fp16 and self._resolved_device == "cuda" else {}
            self._model = CrossEncoder(
                self._model_name,
                device=self._resolved_device,
                model_kwargs=model_kwargs or None,
            )
            if self._max_seq_length is not None:
                self._model.max_seq_length = self._max_seq_length
        return self._model