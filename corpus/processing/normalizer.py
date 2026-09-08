"""Small, conservative normalisation helpers used across corpus stages."""

from __future__ import annotations

import re
import unicodedata


_STANDARD_RE = re.compile(
    r"\b(?P<prefix>IS|SP)\s*(?P<number>\d{1,5})\s*(?P<part>\(\s*(?:Part|Section)\s*\d+\s*\))?\s*(?P<year>[:_]\s*\d{4})?",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    """Normalise Unicode compatibility variants without changing semantic content."""
    return unicodedata.normalize("NFKC", value).replace("\u00a0", " ")


def normalize_standard_number(value: str) -> str | None:
    """Return a stable IS or SP identifier or ``None`` when ``value`` is not an IS/SP reference."""
    match = _STANDARD_RE.search(normalize_text(value))
    if not match:
        return None
    prefix = match.group("prefix").upper()
    part = match.group("part")
    year = match.group("year")
    result = f"{prefix} {match.group('number')}"
    if part:
        inner = re.sub(r"\s+", " ", part.strip()[1:-1]).title()
        result += f" ({inner})"
    if year:
        year_val = re.sub(r"^[:_]\s*", "", year).strip()
        result += f":{year_val}"
    return result

