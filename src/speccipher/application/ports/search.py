"""Standard-library search port definitions for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from speccipher.domain.candidates import Candidate


class VectorIndex(Protocol):
    """Dense index returning candidates in descending similarity rank order."""

    def search(self, query: str, limit: int) -> Sequence[Candidate]:
        """Return ranked candidates with ``dense_score`` populated."""


class LexicalIndex(Protocol):
    """Lexical index returning candidates in BM25 rank order."""

    def search(self, query: str, limit: int) -> Sequence[Candidate]:
        """Return ranked candidates with ``lexical_score`` populated."""
