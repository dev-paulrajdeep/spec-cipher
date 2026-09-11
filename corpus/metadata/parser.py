"""Parsers for document/standard identifiers and dated references."""

from __future__ import annotations

import re

from corpus.processing.normalizer import normalize_document_number


_YEAR_RE = re.compile(
    r"\b(?:IS|SP|ISO|IEC|EN|BS)\s*\d{1,5}"
    r"(?:\s*\(\s*(?:Part|Section)\s*\d+\s*\))?"
    r"(?:\s*[-/]\s*\d+)*"
    r"\s*[:_]\s*(\d{4})",
    re.I,
)
_GENERIC_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


class StandardNumberParser:
    """Parse document/standard numbering formats without guessing omitted identifiers."""

    def parse(self, value: str) -> str | None:
        return normalize_document_number(value)

    def year_from(self, value: str) -> int | None:
        match = _YEAR_RE.search(value)
        if match:
            return int(match.group(1))
        # Conservative fallback: only accept an isolated year near an identifier.
        if normalize_document_number(value):
            years = _GENERIC_YEAR_RE.findall(value)
            if len(years) == 1:
                return int(years[0])
        return None


# Alias for clearer call sites.
DocumentNumberParser = StandardNumberParser
