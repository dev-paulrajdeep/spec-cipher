"""Explainable, standard-level output for a future serving layer."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceItem
from .relationship import RelatedStandard
from .scores import ApplicabilityScore, ConstraintResult, RerankResult


@dataclass(frozen=True)
class Recommendation:
    standard_number: str
    title: str | None
    final_score: float
    rank: int
    constraint_result: ConstraintResult
    rerank_result: RerankResult
    applicability: ApplicabilityScore
    evidence: tuple[EvidenceItem, ...]
    related_standards: tuple[RelatedStandard, ...]
    warnings: tuple[str, ...] = ()
