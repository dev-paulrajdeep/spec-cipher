"""Pure domain retrieval candidates and requirement features.

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import DocumentChunk


@dataclass(frozen=True)
class Candidate:
    """A retrieved chunk candidate with dense, lexical, fusion, and reranking scores.

    Decouples raw retrieval similarity and fused rankings from domain applicability.
    """

    chunk: DocumentChunk
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def similarity_score(self) -> float:
        """First-stage dense similarity score."""
        return self.dense_score if self.dense_score is not None else 0.0

    @property
    def score(self) -> float:
        """Active relevance score (rerank score if available, else dense or RRF)."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.dense_score is not None:
            return self.dense_score
        return self.rrf_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        chunk_raw = data["chunk"]
        chunk = DocumentChunk.from_dict(chunk_raw) if isinstance(chunk_raw, dict) else chunk_raw
        return cls(
            chunk=chunk,
            dense_score=float(data["dense_score"]) if data.get("dense_score") is not None else None,
            lexical_score=float(data["lexical_score"])
            if data.get("lexical_score") is not None
            else None,
            rrf_score=float(data.get("rrf_score", 0.0)),
            rerank_score=float(data["rerank_score"])
            if data.get("rerank_score") is not None
            else None,
        )


@dataclass(frozen=True)
class RequirementFeatures:
    """Parsed requirement lexical and domain concepts."""

    requirement_tokens: frozenset[str] = frozenset()
    product_terms: frozenset[str] = frozenset()
    material_terms: frozenset[str] = frozenset()
    application_terms: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_tokens": sorted(self.requirement_tokens),
            "product_terms": sorted(self.product_terms),
            "material_terms": sorted(self.material_terms),
            "application_terms": sorted(self.application_terms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementFeatures:
        return cls(
            requirement_tokens=frozenset(data.get("requirement_tokens", ())),
            product_terms=frozenset(data.get("product_terms", ())),
            material_terms=frozenset(data.get("material_terms", ())),
            application_terms=frozenset(data.get("application_terms", ())),
        )
