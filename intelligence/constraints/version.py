"""Explicitly requested standard/version checks; absence of a request is not rejection."""

from __future__ import annotations

import re

from corpus.processing.normalizer import normalize_standard_number
from intelligence.models import ConstraintReason


_REFERENCE_RE = re.compile(r"\bIS\s*\d{1,5}(?:\s*\(\s*(?:Part|Section)\s*\d+\s*\))?(?:\s*:\s*\d{4})?", re.I)


def requested_standard_numbers(requirement: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        normalised for match in _REFERENCE_RE.finditer(requirement)
        if (normalised := normalize_standard_number(match.group(0)))
    ))


def version_reasons(candidate_standard: str | None, requirement: str) -> tuple[ConstraintReason, ...]:
    requested = requested_standard_numbers(requirement)
    if not requested or not candidate_standard:
        return ()
    if candidate_standard in requested:
        return ()
    candidate_base = candidate_standard.split(":", 1)[0]
    requested_bases = {item.split(":", 1)[0] for item in requested}
    if candidate_base in requested_bases and all(":" not in item for item in requested):
        return ()
    return (ConstraintReason(
        "explicit_standard_mismatch",
        f"Requirement explicitly requests {', '.join(requested)}, not {candidate_standard}.",
    ),)
