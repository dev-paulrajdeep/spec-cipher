"""Explainable, standard-level output for a future serving layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    status: str | None = None
    version: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete recommendation and its source-backed details."""
        primary_evidence = self.evidence[0] if self.evidence else None
        return {
            "is_code": self.standard_number,
            "standard_number": self.standard_number,
            "title": self.title,
            "final_score": self.final_score,
            "score": self.final_score,
            "reason": (
                f"Retrieved evidence from {primary_evidence.source_file}, "
                f"page {primary_evidence.page_start}."
                if primary_evidence
                else "Recommendation is supported by retrieved source evidence."
            ),
            "source": (
                {
                    "source_file": primary_evidence.source_file,
                    "page_start": primary_evidence.page_start,
                    "page_end": primary_evidence.page_end,
                    "chunk_id": primary_evidence.chunk_id,
                }
                if primary_evidence
                else None
            ),
            "rank": self.rank,
            "status": self.status,
            "version": self.version,
            "constraints": {
                "admissible": self.constraint_result.admissible,
                "reasons": [
                    {"code": reason.code, "message": reason.message}
                    for reason in self.constraint_result.reasons
                ],
            },
            "reranking": {
                "raw_score": self.rerank_result.raw_score,
                "normalized_score": self.rerank_result.normalized_score,
                "rank": self.rerank_result.rank,
                "strategy": self.rerank_result.strategy,
            },
            "applicability": {
                "product_match": self.applicability.product_match,
                "material_match": self.applicability.material_match,
                "application_match": self.applicability.application_match,
                "scope_match": self.applicability.scope_match,
                "requirement_match": self.applicability.requirement_match,
                "overall": self.applicability.overall,
            },
            "evidence": [
                {
                    "standard_number": item.standard_number,
                    "chunk_id": item.chunk_id,
                    "source_file": item.source_file,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "section_number": item.section_number,
                    "section_title": item.section_title,
                    "text": item.text,
                    "relevance_score": item.relevance_score,
                }
                for item in self.evidence
            ],
            "related_standards": [
                {
                    "standard_number": item.standard_number,
                    "relationship_type": item.relationship_type,
                    "source_standard_number": item.source_standard_number,
                    "page_number": item.page_number,
                    "evidence_text": item.evidence_text,
                    "depth": item.depth,
                }
                for item in self.related_standards
            ],
            "warnings": list(self.warnings),
        }
