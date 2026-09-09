"""The single configuration surface for the Intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicabilityWeights:
    product_match: float = 0.25
    material_match: float = 0.15
    application_match: float = 0.20
    scope_match: float = 0.20
    requirement_match: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.product_match, self.material_match, self.application_match,
            self.scope_match, self.requirement_match,
        )
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("Applicability weights must be non-negative with a positive total")


@dataclass(frozen=True)
class IntelligenceConfig:
    """Explicit settings for decision stages; no serving configuration belongs here."""

    reranker: str = "token_overlap"
    cross_encoder_model: str | None = None
    cross_encoder_batch_size: int = 16
    final_rerank_weight: float = 0.35
    minimum_applicability: float = 0.20
    applicability_weights: ApplicabilityWeights = ApplicabilityWeights()
    evidence_limit: int = 3
    relationship_depth: int = 1
    final_top_k: int = 5
    minimum_final_score: float = 0.20
    enable_optional_llm_interpreter: bool = False

    def __post_init__(self) -> None:
        if self.reranker not in {"token_overlap", "cross_encoder"}:
            raise ValueError("reranker must be token_overlap or cross_encoder")
        if self.reranker == "cross_encoder" and not self.cross_encoder_model:
            raise ValueError("cross_encoder_model is required when reranker is cross_encoder")
        if self.cross_encoder_batch_size < 1 or self.evidence_limit < 1 or self.relationship_depth < 0:
            raise ValueError("Batch size, evidence limit, and relationship depth must be valid")
        if self.final_top_k < 1:
            raise ValueError("final_top_k must be positive")
        if not 0 <= self.minimum_final_score <= 1:
            raise ValueError("minimum_final_score must be between zero and one")
        if not 0 <= self.final_rerank_weight <= 1:
            raise ValueError("final_rerank_weight must be between zero and one")
        if not 0 <= self.minimum_applicability <= 1:
            raise ValueError("minimum_applicability must be between zero and one")
