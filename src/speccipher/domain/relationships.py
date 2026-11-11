"""Pure domain relationship entities and taxonomy.

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .identity import StandardId


class RelationshipRole(StrEnum):
    """Formalized relationship taxonomy between standards."""

    PRIMARY = "Primary"
    NORMATIVE_REFERENCE = "Normative Reference"
    TEST_STANDARD = "Test Standard"
    SAFETY_STANDARD = "Safety Standard"
    ALLIED_STANDARD = "Allied Standard"

    @classmethod
    def from_string(cls, val: str) -> RelationshipRole:
        """Map raw or legacy relationship strings to the taxonomy."""
        if not isinstance(val, str):
            return cls.ALLIED_STANDARD
        cleaned = val.strip().lower()
        if "primary" in cleaned:
            return cls.PRIMARY
        if "test" in cleaned:
            return cls.TEST_STANDARD
        if "safety" in cleaned:
            return cls.SAFETY_STANDARD
        if any(token in cleaned for token in ("normative", "reference", "ref")):
            return cls.NORMATIVE_REFERENCE
        return cls.ALLIED_STANDARD


@dataclass(frozen=True)
class StandardRelationship:
    """Source-backed relationship between two standards."""

    source_standard_id: StandardId | None
    target_standard_id: StandardId
    role: RelationshipRole = RelationshipRole.NORMATIVE_REFERENCE
    page_number: int | None = None
    evidence_text: str = ""
    depth: int = 1

    @property
    def standard_number(self) -> str:
        """Backward-compatible alias for target standard code."""
        return self.target_standard_id.code

    @property
    def target_standard_number(self) -> str:
        """Backward-compatible alias for target standard code."""
        return self.target_standard_id.code

    @property
    def source_standard_number(self) -> str | None:
        """Backward-compatible alias for source standard code."""
        return self.source_standard_id.code if self.source_standard_id else None

    @property
    def relationship_type(self) -> str:
        """Backward-compatible alias for role value."""
        return self.role.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_standard_id": self.source_standard_id.to_dict()
            if self.source_standard_id
            else None,
            "target_standard_id": self.target_standard_id.to_dict(),
            "role": self.role.value,
            "page_number": self.page_number,
            "evidence_text": self.evidence_text,
            "depth": self.depth,
            # Backward-compatible fields
            "standard_number": self.standard_number,
            "relationship_type": self.relationship_type,
            "source_standard_number": self.source_standard_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardRelationship:
        target_raw = (
            data.get("target_standard_id")
            or data.get("target_standard_number")
            or data.get("standard_number")
        )
        target_sid = StandardId.from_dict(target_raw)

        source_raw = data.get("source_standard_id") or data.get("source_standard_number")
        source_sid = StandardId.from_dict(source_raw) if source_raw else None

        role_raw = data.get("role") or data.get("relationship_type", "Normative Reference")
        role = RelationshipRole.from_string(str(role_raw))

        return cls(
            source_standard_id=source_sid,
            target_standard_id=target_sid,
            role=role,
            page_number=int(data["page_number"]) if data.get("page_number") is not None else None,
            evidence_text=str(data.get("evidence_text", "")),
            depth=int(data.get("depth", 1)),
        )


# Clean alias
Relationship = StandardRelationship
