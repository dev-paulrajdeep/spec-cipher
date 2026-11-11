"""Pure domain recommendation and categorical confidence entities.

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from .identity import StandardId
from .models import DocumentChunk
from .relationships import StandardRelationship


class ConfidenceLevel(StrEnum):
    """Categorical confidence classification."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass(frozen=True)
class Confidence:
    """Categorical confidence model decoupled from semantic similarity.

    Attributes:
        level: Categorical confidence ("High", "Medium", "Low").
        score: Normalized numeric score in [0.0, 1.0].
        reasons: Machine-readable reason codes justifying the confidence tier.
    """

    level: Literal["High", "Medium", "Low"] | str
    score: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in ("High", "Medium", "Low"):
            raise ValueError(
                f"Confidence level must be 'High', 'Medium', or 'Low', got: {self.level!r}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Confidence score must be in range [0.0, 1.0], got: {self.score}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "score": self.score,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Confidence:
        return cls(
            level=str(data["level"]),
            score=float(data["score"]),
            reasons=tuple(str(r) for r in data.get("reasons", ())),
        )


@dataclass(frozen=True)
class Recommendation:
    """Final, evidence-grounded standard recommendation.

    Fully serializable into modern clean dictionaries and legacy frontend contracts.
    """

    standard_id: StandardId
    title: str
    rank: int
    confidence: Confidence
    similarity_score: float
    applicability_score: float
    evidence: tuple[DocumentChunk, ...] = ()
    related_standards: tuple[StandardRelationship, ...] = ()
    warnings: tuple[str, ...] = ()
    status: str | None = None
    version: str | None = None
    explanation_text: str | None = None
    explanation: dict[str, Any] | None = None
    final_score: float | None = None

    @property
    def standard_number(self) -> str:
        """Backward-compatible standard number alias."""
        return self.standard_id.code

    @property
    def is_code(self) -> str:
        """Backward-compatible IS code identifier alias."""
        return self.standard_id.code

    @property
    def score(self) -> float:
        """Backward-compatible score alias."""
        return self.similarity_score

    def to_dict(self) -> dict[str, Any]:
        """Dual-format serialization supporting both new domain and legacy frontend contracts."""
        primary_evidence = self.evidence[0] if self.evidence else None
        effective_final_score = (
            self.final_score if self.final_score is not None else self.similarity_score
        )

        reason = self.explanation_text or (
            f"Retrieved evidence from {primary_evidence.source_file}, page {primary_evidence.page_start}."
            if primary_evidence
            else "Recommendation is supported by retrieved source evidence."
        )

        source_dict = None
        source_evidence = None
        if primary_evidence:
            source_dict = {
                "source_file": primary_evidence.source_file,
                "page_start": primary_evidence.page_start,
                "page_end": primary_evidence.page_end,
                "chunk_id": primary_evidence.chunk_id,
            }
            source_evidence = {
                "document_title": self.title,
                "edition": self.version,
                "part_volume": None,
                "page_start": primary_evidence.page_start,
                "page_end": primary_evidence.page_end,
                "section_number": primary_evidence.section_number,
                "section_name": primary_evidence.section_title,
            }

        return {
            # Modern domain properties
            "standard_id": self.standard_id.to_dict(),
            "confidence": self.confidence.to_dict(),
            "similarity_score": self.similarity_score,
            "applicability_score": self.applicability_score,
            # Legacy compatibility properties
            "is_code": self.standard_id.code,
            "standard_number": self.standard_id.code,
            "title": self.title,
            "final_score": effective_final_score,
            "score": self.similarity_score,
            "rank": self.rank,
            "reason": reason,
            "explanation_text": reason,
            "source": source_dict,
            "source_evidence": source_evidence,
            "extracted_content": primary_evidence.text if primary_evidence else None,
            "context": (
                f"Retrieved from {primary_evidence.source_file}, page {primary_evidence.page_start}."
                if primary_evidence
                else None
            ),
            "category": {
                "from_source": primary_evidence.source_file if primary_evidence else None,
                "code": primary_evidence.section_number if primary_evidence else None,
                "name": primary_evidence.section_title if primary_evidence else None,
            },
            "category_breadcrumb": (
                f"{primary_evidence.section_number} → {primary_evidence.section_title}"
                if primary_evidence
                and primary_evidence.section_number
                and primary_evidence.section_title
                else None
            ),
            "related_topics": [item.target_standard_id.code for item in self.related_standards],
            "status": self.status,
            "version": self.version,
            "evidence": [item.to_dict() for item in self.evidence],
            "related_standards": [item.to_dict() for item in self.related_standards],
            "warnings": list(self.warnings),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recommendation:
        sid_raw = data.get("standard_id") or data.get("standard_number") or data.get("is_code")
        sid = StandardId.from_dict(sid_raw)

        conf_raw = data.get("confidence")
        if isinstance(conf_raw, dict):
            conf = Confidence.from_dict(conf_raw)
        else:
            conf = Confidence(level="Medium", score=float(data.get("score", 0.5)), reasons=())

        evidence_items = tuple(
            DocumentChunk.from_dict(item) if isinstance(item, dict) else item
            for item in data.get("evidence", ())
        )
        related_items = tuple(
            StandardRelationship.from_dict(item) if isinstance(item, dict) else item
            for item in data.get("related_standards", ())
        )

        return cls(
            standard_id=sid,
            title=str(data.get("title", "")),
            rank=int(data.get("rank", 1)),
            confidence=conf,
            similarity_score=float(data.get("similarity_score", data.get("score", 0.0))),
            applicability_score=float(data.get("applicability_score", 1.0)),
            evidence=evidence_items,
            related_standards=related_items,
            warnings=tuple(str(w) for w in data.get("warnings", ())),
            status=data.get("status"),
            version=data.get("version"),
            explanation_text=data.get("explanation_text") or data.get("reason"),
            explanation=data.get("explanation"),
            final_score=float(data["final_score"]) if data.get("final_score") is not None else None,
        )
