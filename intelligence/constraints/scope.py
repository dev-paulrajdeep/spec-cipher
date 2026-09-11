"""Conservative scope exclusions based only on explicit corpus scope wording."""

from __future__ import annotations

import re

from intelligence.models import ConstraintReason
from intelligence.text import tokenize


_EXCLUSION_RE = re.compile(r"\b(?:does\s+not\s+apply\s+to|excluding)\s+([^.;\n]+)", re.I)
_GENERIC_REQUIREMENT_TERMS = frozenset({
    "applicable", "compliance", "equipment", "general", "industrial",
    "purpose", "requirement", "requirements", "service", "standard",
    "standards", "system", "technical", "commercial", "building",
    "facility", "facilities", "plant", "installation", "installations",
    "item", "items", "unit", "units", "device", "devices",
})
_PRODUCT_TERMS = frozenset({
    "pump", "pumps", "centrifugal", "impeller", "volute", "motor", "motors",
    "induction", "efficiency", "valve", "valves", "compressor", "boiler",
    "transformer",
})


def _fold_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _folded(tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(_fold_token(token) for token in tokens)


def _primary_requirement_span(requirement: str) -> str:
    """Use the head clause before 'for' when present so context nouns do not dominate."""
    lowered = " ".join(requirement.split())
    match = re.search(r"\s+for\s+", lowered, flags=re.IGNORECASE)
    if match:
        return lowered[: match.start()]
    return lowered


def title_product_reasons(requirement: str, title: str | None) -> tuple[ConstraintReason, ...]:
    """Reject when a titled product family has no overlap with the requirement's primary product span."""
    if not title:
        return ()
    title_products = tokenize(title).intersection(_PRODUCT_TERMS)
    if not title_products:
        return ()
    primary_products = tokenize(_primary_requirement_span(requirement)).intersection(_PRODUCT_TERMS)
    if not primary_products:
        return (
            ConstraintReason(
                "insufficient_product_alignment",
                "Requirement does not name a product family aligned with the candidate standard title.",
            ),
        )
    if not _folded(primary_products).intersection(_folded(title_products)):
        return (
            ConstraintReason(
                "insufficient_product_alignment",
                "Requirement product focus does not align with the candidate standard title.",
            ),
        )
    return ()


def scope_reasons(requirement: str, scope_text: str | None) -> tuple[ConstraintReason, ...]:
    """Reject only clear 'does not apply' overlaps; unknown scope remains admissible."""
    query_tokens = tokenize(requirement)
    reasons: list[ConstraintReason] = []
    # A generic procurement phrase is not sufficient evidence to name a
    # standard.  This is independent of semantic similarity and avoids a
    # one-chunk corpus turning broad wording into a confident recommendation.
    if not query_tokens.difference(_GENERIC_REQUIREMENT_TERMS):
        reasons.append(ConstraintReason(
            "insufficient_requirement_specificity",
            "Requirement does not identify a product, material, application, or other specific scope term.",
        ))
    if not scope_text:
        return tuple(reasons)
    for match in _EXCLUSION_RE.finditer(scope_text):
        excluded = tokenize(match.group(1))
        overlap = query_tokens.intersection(excluded)
        if overlap:
            reasons.append(ConstraintReason(
                "explicit_scope_exclusion",
                f"Scope explicitly excludes: {match.group(1).strip()}.",
            ))
    return tuple(reasons)
