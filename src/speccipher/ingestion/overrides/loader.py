"""Auditable operator override loader and validation (Layer 4 Ingestion)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from speccipher.domain.errors import OverrideValidationError
from speccipher.domain.identity import StandardId


@dataclass(frozen=True)
class OperatorOverride:
    """Represents an auditable operator override for unextractable standard identities."""

    operator_id: str
    timestamp: str
    reason: str
    asserted_identity: StandardId
    evidence_sources: tuple[str, ...] = ()
    source_filename: str | None = None

    @property
    def operator(self) -> str:
        """Backward-compatible alias for the original override field."""
        return self.operator_id

    @property
    def rationale(self) -> str:
        """Backward-compatible alias for the original override field."""
        return self.reason

    @property
    def asserted_standard(self) -> StandardId:
        """Backward-compatible alias for the original override field."""
        return self.asserted_identity

    def to_dict(self) -> dict[str, Any]:
        result = {
            "asserted_identity": self.asserted_identity.to_dict(),
            "operator_id": self.operator_id,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "evidence_sources": list(self.evidence_sources),
            # Existing ingestion tools consume these legacy names. Keep them
            # until their persisted override files have migrated to schema 2.
            "operator": self.operator_id,
            "rationale": self.reason,
            "asserted_standard": self.asserted_identity.to_dict(),
        }
        if self.source_filename is not None:
            result["source_filename"] = self.source_filename
        return result


def validate_operator_override(override_data: dict[str, Any]) -> OperatorOverride:
    """Validate operator override dictionary against required audit fields."""
    if not isinstance(override_data, dict):
        raise OverrideValidationError("Operator override must be an object")

    new_format = "asserted_identity" in override_data
    identity_key = "asserted_identity" if new_format else "asserted_standard"
    operator_key = "operator_id" if new_format else "operator"
    reason_key = "reason" if new_format else "rationale"
    required_fields = (operator_key, "timestamp", reason_key, identity_key)
    if not new_format:
        required_fields = (*required_fields, "source_filename")
    missing = [field for field in required_fields if not override_data.get(field)]
    if missing:
        raise OverrideValidationError(
            f"Missing required audit or identity fields in operator override: {', '.join(missing)}"
        )

    for field in (operator_key, reason_key):
        value = override_data[field]
        if not isinstance(value, str) or not value.strip():
            raise OverrideValidationError(f"Operator override {field} must be a non-empty string")
    source_filename = override_data.get("source_filename")
    if source_filename is not None and (
        not isinstance(source_filename, str) or not source_filename.strip()
    ):
        raise OverrideValidationError("Operator override source_filename must be a non-empty string")

    raw_evidence = override_data.get("evidence_sources", [])
    if not isinstance(raw_evidence, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_evidence
    ):
        raise OverrideValidationError("Operator override evidence_sources must be an array of strings")
    if new_format and not raw_evidence:
        raise OverrideValidationError("Operator override evidence_sources must not be empty")

    timestamp = override_data["timestamp"]
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise OverrideValidationError("Operator override timestamp must be an ISO-8601 string")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise OverrideValidationError("Operator override timestamp must be ISO-8601") from exc
    if parsed_timestamp.tzinfo is None:
        raise OverrideValidationError("Operator override timestamp must include a timezone")

    raw_std = override_data[identity_key]
    try:
        if isinstance(raw_std, StandardId):
            std_id = raw_std
        elif isinstance(raw_std, str):
            std_id = StandardId.parse(raw_std)
        elif isinstance(raw_std, dict):
            std_id = StandardId.from_dict(raw_std)
            supplied_code = raw_std.get("code")
            if supplied_code:
                parsed_code = StandardId.parse(supplied_code)
                if parsed_code != StandardId(
                    family=std_id.family,
                    number=std_id.number,
                    part=std_id.part,
                    year=std_id.year,
                ):
                    raise ValueError("asserted_standard code does not match its identity fields")
        else:
            raise ValueError(f"{identity_key} must be a StandardId or identity object")
    except (KeyError, TypeError, ValueError) as exc:
        raise OverrideValidationError(f"Invalid {identity_key}: {exc}") from exc

    return OperatorOverride(
        source_filename=source_filename.strip() if source_filename is not None else None,
        operator_id=override_data[operator_key].strip(),
        timestamp=timestamp.strip(),
        reason=override_data[reason_key].strip(),
        asserted_identity=std_id,
        evidence_sources=tuple(item.strip() for item in raw_evidence),
    )
