"""Separate score types prevent semantic, rerank, and fit scores being conflated."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintReason:
    code: str
    message: str


@dataclass(frozen=True)
class ConstraintResult:
    admissible: bool
    reasons: tuple[ConstraintReason, ...] = ()


@dataclass(frozen=True)
class RerankResult:
    """Cross/query-passage relevance score; it is not applicability."""

    raw_score: float
    normalized_score: float
    rank: int
    strategy: str


@dataclass(frozen=True)
class ApplicabilityScore:
    """Degree-of-fit decomposition among candidates that passed hard constraints."""

    product_match: float
    material_match: float
    application_match: float
    scope_match: float
    requirement_match: float
    overall: float

    def __post_init__(self) -> None:
        values = (
            self.product_match, self.material_match, self.application_match,
            self.scope_match, self.requirement_match, self.overall,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("Applicability scores must be within [0, 1]")
