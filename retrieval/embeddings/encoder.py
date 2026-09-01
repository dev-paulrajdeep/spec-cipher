"""Protocol isolating semantic embedding generation from vector indexing."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingEncoder(Protocol):
    """A deterministic encoder configuration used to create and query one index."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalize_embeddings(self) -> bool: ...

    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_batch(self, texts: list[str]) -> np.ndarray: ...
