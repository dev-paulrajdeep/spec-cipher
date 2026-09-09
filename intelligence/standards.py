"""Canonical handling for explicitly identified Indian Standards."""

from __future__ import annotations

import re


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
        code += f" (Part {" ".join(part.split())})"
    if year:
        code += f":{year}"
    return code