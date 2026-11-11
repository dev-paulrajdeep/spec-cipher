"""Categorical confidence calculation (Layer 4 Decision)."""

from __future__ import annotations

from typing import Any

from speccipher.domain.recommendation import Confidence


def calculate_categorical_confidence(
    evidence_chunks: list[dict[str, Any]],
    is_eligible: bool,
    margin: float = 0.2,
) -> Confidence:
    """Evaluate categorical confidence and machine-readable reasons."""
    reasons: list[str] = []
    if is_eligible:
        reasons.append("resolved_identity")
        reasons.append("status_current")

    has_scope = any("scope" in (c.get("section_title") or "").lower() for c in evidence_chunks)
    if has_scope:
        reasons.append("scope_clause_grounded")

    if len(evidence_chunks) >= 2:
        reasons.append("multi_chunk_evidence")
    else:
        reasons.append("single_chunk_evidence")

    if margin > 0.15:
        reasons.append("distinct_score_margin")
    else:
        reasons.append("narrow_score_margin")

    if is_eligible and len(evidence_chunks) >= 2 and has_scope and margin > 0.15:
        return Confidence(level="High", score=0.90, reasons=tuple(reasons))
    elif is_eligible and len(evidence_chunks) >= 1:
        return Confidence(level="Medium", score=0.65, reasons=tuple(reasons))
    else:
        return Confidence(level="Low", score=0.35, reasons=tuple(reasons))
