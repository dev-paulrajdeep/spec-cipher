"""Lazy adapter around a pretrained Sentence Transformers model."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEncoder:
    """Batching, normalisation, and model identity around SentenceTransformer.

    The model is loaded lazily so corpus/index tests can inject a deterministic
    synthetic encoder and never download model weights.
    """

    def __init__(self, model_name: str, batch_size: int = 32, normalize_embeddings: bool = True) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    @property
    def dimension(self) -> int:
        dimension = self._load_model().get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("Sentence Transformer did not report an embedding dimension")
        return int(dimension)

    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed an empty query")
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input texts must all be non-empty strings")
        vectors = self._load_model().encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for SentenceTransformerEncoder; "
                    "install the retrieval dependencies."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model
