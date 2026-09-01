"""Parsers for Indian Standard identifiers and dated references."""

from __future__ import annotations

import re

from corpus.processing.normalizer import normalize_standard_number


_STANDARD_WITH_YEAR_RE = re.compile(r"\bIS\s*\d{1,5}(?:\s*\(\s*(?:Part|Section)\s*\d+\s*\))?\s*:\s*(\d{4})", re.I)


class StandardNumberParser:
    """Parse IS numbering formats without guessing omitted identifiers."""

    def parse(self, value: str) -> str | None:
        return normalize_standard_number(value)

    def year_from(self, value: str) -> int | None:
        match = _STANDARD_WITH_YEAR_RE.search(value)
        return int(match.group(1)) if match else None
