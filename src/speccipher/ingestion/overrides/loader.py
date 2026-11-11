"""Auditable operator override loader and validation (Layer 4 Ingestion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from speccipher.domain.errors import OverrideValidationError
from speccipher.domain.identity import StandardId


@dataclass(frozen=True)
class OperatorOverride:
    """Represents an auditable operator override for unextractable standard identities."""

    source_filename: str
    operator: str
    timestamp: str
    rationale: str
    asserted_standard: StandardId

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "operator": self.operator,
            "timestamp": self.timestamp,
            "rationale": self.rationale,
            "asserted_standard": self.asserted_standard.to_dict(),
        }


def validate_operator_override(override_data: dict[str, Any]) -> OperatorOverride:
    """Validate operator override dictionary against required audit fields."""
    required_fields = ["source_filename", "operator", "timestamp", "rationale"]
    missing = [f for f in required_fields if not override_data.get(f)]
    if missing:
        raise OverrideValidationError(
            f"Missing required audit fields in operator override: {', '.join(missing)}"
        )

    raw_std = override_data.get("asserted_standard", {})
    if isinstance(raw_std, StandardId):
        std_id = raw_std
    elif isinstance(raw_std, dict):
        std_id = StandardId(
            family=raw_std.get("family", "IS"),
            number=str(raw_std.get("number", "0")),
            part=str(raw_std.get("part")) if raw_std.get("part") is not None else None,
            year=int(raw_std.get("year")) if raw_std.get("year") is not None else None,
            code=raw_std.get("code") or "UNKNOWN",
        )
    else:
        std_id = StandardId(family="IS", number="0", code="UNKNOWN")

    return OperatorOverride(
        source_filename=str(override_data["source_filename"]),
        operator=str(override_data["operator"]),
        timestamp=str(override_data["timestamp"]),
        rationale=str(override_data["rationale"]),
        asserted_standard=std_id,
    )
