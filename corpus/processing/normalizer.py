"""Small, conservative normalisation helpers used across corpus stages."""

from __future__ import annotations

import re
import unicodedata


_STANDARD_RE = re.compile(
    r"\bIS\s*(?P<number>\d{1,5})\s*(?P<part>\(\s*(?:Part|Section)\s*\d+\s*\))?\s*(?P<year>:\s*\d{4})?",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    """Normalise Unicode compatibility variants without changing semantic content."""
    return unicodedata.normalize("NFKC", value).replace("\u00a0", " ")


def normalize_standard_number(value: str) -> str | None:
    """Return a stable IS identifier or ``None`` when ``value`` is not an IS reference."""
    match = _STANDARD_RE.search(normalize_text(value))
    if not match:
        return None
    part = match.group("part")
    year = match.group("year")
    result = f"IS {match.group('number')}"
    if part:
        inner = re.sub(r"\s+", " ", part.strip()[1:-1]).title()
        result += f" ({inner})"
    if year:
        result += ":" + year.split(":", 1)[1].strip()
    return result
