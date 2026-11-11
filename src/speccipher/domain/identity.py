"""Pure domain standard identity entities and normalization helpers.

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_FAMILY_RE = re.compile(
    r"^\s*(?P<family>ISO|IEC|IS|SP|EN|BS|IRC)\s*"
    r"(?P<number>\d{1,6})"
    r"(?:[\s\-_]*(?:\(|Part|Section|Volume|\/)*\s*(?P<part>\d+[A-Za-z]?|[A-Za-z]+\s*\d*|\d+[-/]\d+)\s*\)?)?"
    r"(?:\s*[:_/\-]\s*(?P<year>\d{4}))?\s*$",
    re.IGNORECASE,
)

_ASTM_RE = re.compile(
    r"^\s*ASTM\s*(?P<number>[A-Z]\s*\d{1,5})"
    r"(?:\s*[:_/\-]\s*(?P<year>\d{2,4}))?\s*$",
    re.IGNORECASE,
)

_GENERIC_ID_RE = re.compile(r"^[A-Z]{2,12}(?:[-_][A-Z0-9]{1,12}){1,5}$")


def format_standard_code(
    family: str,
    number: str,
    part: str | None = None,
    year: int | None = None,
) -> str:
    """Format canonical string representation of a standard identifier."""
    fam = family.strip().upper()
    num = number.strip()
    if fam == "GENERIC":
        return num
    if fam == "ASTM":
        code = f"ASTM {num}"
        if year is not None:
            code += f":{year}"
        return code

    code = f"{fam} {num}"
    if part:
        cleaned_part = str(part).strip()
        if cleaned_part.lower().startswith("part "):
            cleaned_part = cleaned_part[5:].strip()
        elif cleaned_part.lower().startswith("(part ") and cleaned_part.endswith(")"):
            cleaned_part = cleaned_part[6:-1].strip()
        code += f" (Part {cleaned_part})"
    if year is not None:
        code += f":{year}"
    return code


@dataclass(frozen=True)
class StandardId:
    """Immutable canonical identity of an engineering standard or specification.

    Attributes:
        family: Standard organization / family (e.g., 'IS', 'SP', 'ISO', 'ASTM').
        number: Base numeric or alphanumeric identifier (e.g., '456', '7', 'C150').
        part: Optional part, section, or volume specifier (e.g., '1', '3A').
        year: Optional revision or publication year (e.g., 2000, 2026).
        code: Canonical unified identifier string (e.g., 'IS 456:2000', 'SP 7:2026').
    """

    family: str
    number: str
    part: str | None = None
    year: int | None = None
    code: str = ""

    def __post_init__(self) -> None:
        if not self.family or not self.family.strip():
            raise ValueError("StandardId family must be a non-empty string")
        if not self.number or not self.number.strip():
            raise ValueError("StandardId number must be a non-empty string")
        if self.year is not None and not (1800 <= self.year <= 2100):
            raise ValueError(f"StandardId year out of valid range: {self.year}")

        normalized_family = self.family.strip().upper()
        normalized_number = self.number.strip()
        object.__setattr__(self, "family", normalized_family)
        object.__setattr__(self, "number", normalized_number)

        if not self.code:
            canonical_code = format_standard_code(
                normalized_family,
                normalized_number,
                self.part,
                self.year,
            )
            object.__setattr__(self, "code", canonical_code)

    def __str__(self) -> str:
        return self.code

    @classmethod
    def parse(cls, value: str) -> StandardId:
        """Parse a standard identifier string into a canonical StandardId.

        Supports standard patterns such as:
            - 'IS 456:2000', 'IS456', 'IS 456 : 2000'
            - 'IS 875 (Part 3):2015', 'IS 3370-1:2009'
            - 'SP 7:2026', 'SP 7 (Volume 1):2026'
            - 'ASTM C150:2020'
            - 'ISO 9001:2015', 'IEC 60034-1:2017'
            - 'BOOK-18-ELECTRONICS-AND' (generic catalog identifier)

        Raises:
            ValueError: If the string cannot be parsed into a valid standard ID.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Cannot parse StandardId from invalid input: {value!r}")

        raw = value.strip()
        fam_match = _FAMILY_RE.match(raw)
        if fam_match:
            d = fam_match.groupdict()
            year_val = int(d["year"]) if d["year"] else None
            part_val = d["part"]
            if part_val:
                part_val = re.sub(
                    r"^(?:Part|Section|Volume|\()\s*", "", part_val, flags=re.I
                ).rstrip(")")
                part_val = part_val.strip()
            return cls(
                family=d["family"].upper(),
                number=d["number"],
                part=part_val or None,
                year=year_val,
            )

        astm_match = _ASTM_RE.match(raw)
        if astm_match:
            d = astm_match.groupdict()
            year_val = int(d["year"]) if d["year"] else None
            number_val = re.sub(r"\s+", "", d["number"].upper())
            return cls(
                family="ASTM",
                number=number_val,
                part=None,
                year=year_val,
            )

        if _GENERIC_ID_RE.match(raw.upper()):
            return cls(
                family="GENERIC",
                number=raw.upper(),
                part=None,
                year=None,
                code=raw.upper(),
            )

        raise ValueError(f"Cannot parse standard identity from: {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a pure Python JSON-compatible dictionary."""
        return {
            "family": self.family,
            "number": self.number,
            "part": self.part,
            "year": self.year,
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> StandardId:
        """Deserialize from a dictionary or string."""
        if isinstance(data, str):
            return cls.parse(data)
        return cls(
            family=str(data["family"]),
            number=str(data["number"]),
            part=data.get("part"),
            year=int(data["year"]) if data.get("year") is not None else None,
            code=str(data.get("code") or ""),
        )


@dataclass(frozen=True)
class StandardStub:
    """Representation of an uningested or referenced standard.

    Invariants:
        - `is_stub` is always True.
        - `is_eligible` is always False (cannot become a primary recommendation).
    """

    standard_id: StandardId
    title: str | None = None
    referenced_by_chunk_ids: tuple[str, ...] = ()
    is_stub: bool = True
    is_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.is_stub:
            object.__setattr__(self, "is_stub", True)
        if self.is_eligible:
            object.__setattr__(self, "is_eligible", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standard_id": self.standard_id.to_dict(),
            "title": self.title,
            "referenced_by_chunk_ids": list(self.referenced_by_chunk_ids),
            "is_stub": self.is_stub,
            "is_eligible": self.is_eligible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardStub:
        sid_raw = data["standard_id"]
        sid = StandardId.from_dict(sid_raw) if isinstance(sid_raw, (dict, str)) else sid_raw
        return cls(
            standard_id=sid,
            title=data.get("title"),
            referenced_by_chunk_ids=tuple(data.get("referenced_by_chunk_ids", ())),
            is_stub=True,
            is_eligible=False,
        )
