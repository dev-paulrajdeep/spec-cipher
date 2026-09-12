"""Canonical handling for recommendation document identities."""

from __future__ import annotations

import re

from corpus.processing.normalizer import normalize_document_number


_IS_CODE_RE = re.compile(
    r"\bIS\s*(?P<number>\d{1,6})"
    r"(?:\s*\(\s*Part\s*(?P<part>[^)]+?)\s*\))?"
    r"(?:\s*:\s*(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)


def canonical_is_code(value: str | None) -> str | None:
    """Return a normalized IS identifier, or None when no code is evidenced."""
    if not isinstance(value, str):
        return None
    match = _IS_CODE_RE.search(value)
    if not match:
        return None
    number = match.group("number")
    part = match.group("part")
    year = match.group("year")
    code = f"IS {number}"
    if part:
        normalized_part = " ".join(part.split())
        code += f" (Part {normalized_part})"
    if year:
        code += f":{year}"
    return code


def canonical_recommendation_id(value: str | None) -> str | None:
    """Return a stable recommendation ID for independently indexed documents.

    Prefers normalized IS codes. Otherwise accepts normalized document numbers
    already present on corpus identities. Does not invent identifiers from prose.
    """
    is_code = canonical_is_code(value)
    if is_code:
        return is_code
    if not isinstance(value, str) or not value.strip():
        return None
    return normalize_document_number(value)