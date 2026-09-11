"""Lazy adapter around a pretrained Sentence Transformers model."""

from __future__ import annotations

import logging
import os
from threading import RLock

import numpy as np

logger = logging.getLogger(__name__)


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
        self._resolved_device = device
        self._model_lock = RLock()

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
        try:
            vectors = self._encode_with_loaded_model(texts)
        except Exception:
            if self.device != "cuda":
                raise
            logger.warning("CUDA embedding inference failed; retrying with CPU", exc_info=True)
            self._fallback_to_cpu()
            vectors = self._encode_with_loaded_model(texts)
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
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch and sentence-transformers are required for SentenceTransformerEncoder; "
                    "install the retrieval dependencies."
                ) from exc

            preferred_device = "cpu"
            if self._device != "cpu":
                try:
                    if torch.cuda.is_available():
                        preferred_device = "cuda"
                except Exception:
                    logger.warning("CUDA availability check failed; using CPU", exc_info=True)

            if self._device == "cuda" and preferred_device != "cuda" and not self._allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA was requested for embeddings but is unavailable; set "
                    "SPEC_CIPHER_ALLOW_CPU_FALLBACK=1 to permit CPU fallback"
                )
            try:
                self._model = self._create_model(SentenceTransformer, torch, preferred_device)
                self._resolved_device = preferred_device
            except Exception:
                if preferred_device != "cuda" or not self._allow_cpu_fallback:
                    raise
                logger.warning("CUDA model initialization failed; retrying with CPU", exc_info=True)
                self._clear_model(torch)
                self._model = self._create_model(SentenceTransformer, torch, "cpu")
                self._resolved_device = "cpu"
            return self._model

    def _create_model(self, sentence_transformer, torch, device: str):
        model_kwargs = {"torch_dtype": torch.float16} if self._use_fp16 and device == "cuda" else {}
        kwargs = {"device": device, "model_kwargs": model_kwargs or None}
        if os.getenv("SPEC_CIPHER_HF_OFFLINE", "0") == "1" or os.getenv("SPEC_CIPHER_ALLOW_MODEL_DOWNLOAD", "0") != "1":
            # This is a SentenceTransformer constructor option, not a Hugging
            # Face model kwarg. Cached serving is the safe default: model
            # acquisition is an explicit setup action, never a surprise at
            # backend startup.
            kwargs["local_files_only"] = True
        model = sentence_transformer(self._model_name, **kwargs)
        if self._max_seq_length is not None:
            model.max_seq_length = self._max_seq_length
        return model

    def _encode_with_loaded_model(self, texts: list[str]):
        return self._load_model().encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )

    def _fallback_to_cpu(self) -> None:
        with self._model_lock:
            if not self._allow_cpu_fallback:
                raise RuntimeError("CUDA embedding inference failed and CPU fallback is disabled")
            import torch
            from sentence_transformers import SentenceTransformer

            self._clear_model(torch)
            self._model = self._create_model(SentenceTransformer, torch, "cpu")
            self._resolved_device = "cpu"

    def _clear_model(self, torch) -> None:
        self._model = None
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cleanup failed while switching to CPU", exc_info=True)

    def release(self) -> None:
        """Release GPU/memory resources held by the loaded model."""
        with self._model_lock:
            if self._model is not None:
                try:
                    import torch
                    self._clear_model(torch)
                except ImportError:
                    self._model = None

