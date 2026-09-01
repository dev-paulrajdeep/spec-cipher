"""Conservative scope exclusions based only on explicit corpus scope wording."""

from __future__ import annotations

import re

from intelligence.models import ConstraintReason
from intelligence.text import tokenize


_EXCLUSION_RE = re.compile(r"\b(?:does\s+not\s+apply\s+to|excluding)\s+([^.;\n]+)", re.I)


def scope_reasons(requirement: str, scope_text: str | None) -> tuple[ConstraintReason, ...]:
    """Reject only clear 'does not apply' overlaps; unknown scope remains admissible."""
    if not scope_text:
        return ()
    query_tokens = tokenize(requirement)
    reasons: list[ConstraintReason] = []
    for match in _EXCLUSION_RE.finditer(scope_text):
        excluded = tokenize(match.group(1))
        overlap = query_tokens.intersection(excluded)
        if overlap:
            reasons.append(ConstraintReason(
                "explicit_scope_exclusion",
                f"Scope explicitly excludes: {match.group(1).strip()}.",
            ))
    return tuple(reasons)
