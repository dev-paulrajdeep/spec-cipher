"""Wrap retrieval output while preserving every intermediate decision result."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.models import RetrievalResult
from .scores import ApplicabilityScore, ConstraintResult, RerankResult


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: RetrievalResult
    constraint_result: ConstraintResult
    rerank_result: RerankResult | None = None
    applicability: ApplicabilityScore | None = None
    final_score: float | None = None
    warnings: tuple[str, ...] = ()
