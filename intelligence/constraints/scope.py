"""Conservative scope exclusions based only on explicit corpus scope wording."""

from __future__ import annotations

import re

from intelligence.models import ConstraintReason
from intelligence.text import tokenize


_EXCLUSION_RE = re.compile(r"\b(?:does\s+not\s+apply\s+to|excluding)\s+([^.;\n]+)", re.I)
_GENERIC_REQUIREMENT_TERMS = frozenset({
    "applicable", "compliance", "equipment", "general", "industrial",
    "purpose", "requirement", "requirements", "service", "standard",
    "standards", "system", "technical",
})


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
