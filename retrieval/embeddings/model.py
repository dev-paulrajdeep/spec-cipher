"""Lazy adapter around a pretrained Sentence Transformers model."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEncoder:
    """Batching, normalisation, and model identity around SentenceTransformer.

    The model is loaded lazily so corpus/index tests can inject a deterministic
    synthetic encoder and never download model weights.
    """

    backend_name = "sentence-transformers"

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        *,
        device: str = "auto",
        allow_cpu_fallback: bool = False,
        max_seq_length: int | None = None,
        embedding_dimension: int | None = None,
        use_fp16: bool = True,
        model_version: str = "main",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        if max_seq_length is not None and max_seq_length < 1:
            raise ValueError("max_seq_length must be positive when provided")
        if embedding_dimension is not None and embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive when provided")
        if not isinstance(use_fp16, bool):
            raise ValueError("use_fp16 must be a boolean")
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        self._model_name = model_name
        self._model_version = model_version
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._device = device
        self._allow_cpu_fallback = allow_cpu_fallback
        self._max_seq_length = max_seq_length
        self._embedding_dimension = embedding_dimension
        self._use_fp16 = use_fp16
        self._model = None

    @property
    def configured_dimension(self) -> int | None:
        """Return a persisted dimension without loading model weights."""
        return self._embedding_dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        return self._resolved_device if self._model is not None else self._device

    @property
    def dimension(self) -> int:
        model = self._load_model()
        getter = getattr(model, "get_embedding_dimension", None)
        dimension = getter() if getter is not None else model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("Sentence Transformer did not report an embedding dimension")
        dimension = int(dimension)
        if self._embedding_dimension is not None and dimension != self._embedding_dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: configured {self._embedding_dimension}, model returned {dimension}"
            )
        return dimension

    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed an empty query")
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self._embedding_dimension or (self.dimension if self._model is not None else 0)
            return np.empty((0, dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input texts must all be non-empty strings")
        vectors = self._load_model().encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        values = np.ascontiguousarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape != (len(texts), self.dimension):
            raise RuntimeError(f"Embedding backend returned an incompatible shape: {values.shape}")
        if self._normalize_embeddings:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms == 0) or not np.isfinite(norms).all():
                raise RuntimeError("Embedding backend returned a zero or non-finite vector")
            values = np.ascontiguousarray(values / norms, dtype=np.float32)
        return values

    def _load_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch and sentence-transformers are required for SentenceTransformerEncoder; "
                    "install the retrieval dependencies."
                ) from exc
            cuda_available = bool(torch.cuda.is_available())
            if self._device == "cpu":
                self._resolved_device = "cpu"
            elif cuda_available:
                self._resolved_device = "cuda"
            elif self._allow_cpu_fallback:
                self._resolved_device = "cpu"
            else:
                raise RuntimeError(
                    "CUDA is unavailable for the configured embedding device; "
                    "set allow_cpu_fallback=True or embedding_device='cpu' explicitly"
                )
            model_kwargs = {"torch_dtype": torch.float16} if self._use_fp16 and self._resolved_device == "cuda" else {}
            self._model = SentenceTransformer(
                self._model_name,
                device=self._resolved_device,
                model_kwargs=model_kwargs or None,
            )
            if self._max_seq_length is not None:
                self._model.max_seq_length = self._max_seq_length
        return self._model

    def release(self) -> None:
        """Release GPU/memory resources held by the loaded model."""
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

