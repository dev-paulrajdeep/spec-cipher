"""Small, conservative normalisation helpers used across corpus stages."""

from __future__ import annotations

import re
import unicodedata


# Family-prefixed standards identifiers (year/part optional).
_FAMILY_RE = re.compile(
    r"\b(?P<prefix>ISO|IEC|IS|SP|EN|BS)\s*"
    r"(?P<number>\d{1,5})\s*"
    r"(?P<part>\(\s*(?:Part|Section)\s*\d+\s*\))?"
    r"(?P<suffix>(?:\s*[-/]\s*\d+)*)?"
    r"\s*(?P<year>[:_]\s*\d{4})?",
    re.IGNORECASE,
)
_ASTM_RE = re.compile(
    r"\bASTM\s*(?P<code>[A-Z]\s*\d{1,5})(?P<year>\s*[-/]\s*\d{2,4})?",
    re.IGNORECASE,
)
# Structured document ids such as TECH-1001, DOC-2026-001, STD-A.
_GENERIC_ID_RE = re.compile(
    r"\b(?P<code>[A-Z]{2,12}(?:[-_][A-Z0-9]{1,12}){1,3})\b"
)
_GENERIC_BLOCKLIST = frozenset(
    {
        "PART",
        "SECTION",
        "ANNEX",
        "ANNEXURE",
        "APPENDIX",
        "TABLE",
        "FIGURE",
        "CHAPTER",
        "VOLUME",
        "PAGE",
        "CLAUSE",
        "NOTE",
        "ITEM",
        "TYPE",
        "CLASS",
        "GRADE",
        "TEMP",
        "TMP",
        "COPY",
        "SCAN",
        "FILE",
        "DOWNLOAD",
        "UNTITLED",
        "DOCUMENT",
        "IMAGE",
        "EXPORT",
        "CEMENT",
        "PRODUCT",
        "CATEGORY",
    }
)


def normalize_text(value: str) -> str:
    """Normalise Unicode compatibility variants without changing semantic content."""
    return unicodedata.normalize("NFKC", value).replace("\u00a0", " ")


def normalize_document_number(value: str) -> str | None:
    """Return a stable document/standard number, or ``None`` when none is evidenced.

    Supports common family prefixes (IS/SP/ISO/IEC/EN/BS/ASTM) and conservative
    structured identifiers. Does not invent identifiers from prose filenames.
    """
    text = normalize_text(value).strip()
    if not text:
        return None

    # Prefer family-prefixed forms even when filenames use underscores/hyphens.
    family_candidates = (text, re.sub(r"[_\-]+", " ", text))
    for candidate in family_candidates:
        family = _FAMILY_RE.search(candidate)
        if family:
            prefix = family.group("prefix").upper()
            result = f"{prefix} {family.group('number')}"
            part = family.group("part")
            if part:
                inner = re.sub(r"\s+", " ", part.strip()[1:-1]).title()
                result += f" ({inner})"
            suffix = family.group("suffix")
            if suffix:
                cleaned = re.sub(r"\s+", "", suffix)
                result += cleaned
            year = family.group("year")
            if year:
                year_val = re.sub(r"^[:_]\s*", "", year).strip()
                result += f":{year_val}"
            return result

    astm = _ASTM_RE.search(text) or _ASTM_RE.search(re.sub(r"[_\-]+", " ", text))
    if astm:
        code = re.sub(r"\s+", "", astm.group("code").upper())
        result = f"ASTM {code}"
        year = astm.group("year")
        if year:
            result += re.sub(r"\s+", "", year)
        return result

    generic = _GENERIC_ID_RE.search(text.upper().replace("_", "-"))
    if generic:
        code = generic.group("code").upper().replace("_", "-")
        head = code.split("-", 1)[0]
        if head in _GENERIC_BLOCKLIST:
            return None
        # Require at least one digit or a single-letter trailing token (STD-A).
        if not re.search(r"\d", code) and not re.search(r"-[A-Z]$", code):
            return None
        return code
    return None


def normalize_standard_number(value: str) -> str | None:
    """Backward-compatible alias for :func:`normalize_document_number`."""
    return normalize_document_number(value)


def find_document_numbers(text: str) -> tuple[str, ...]:
    """Return ordered unique document numbers found in ``text``."""
    found: list[str] = []
    seen: set[str] = set()
    normalized = normalize_text(text)
    for match in _FAMILY_RE.finditer(normalized):
        value = normalize_document_number(match.group(0))
        if value and value not in seen:
            seen.add(value)
            found.append(value)
    for match in _ASTM_RE.finditer(normalized):
        value = normalize_document_number(match.group(0))
        if value and value not in seen:
            seen.add(value)
            found.append(value)
    for match in _GENERIC_ID_RE.finditer(normalized.upper()):
        value = normalize_document_number(match.group(0))
        if value and value not in seen:
            seen.add(value)
            found.append(value)
    return tuple(found)
