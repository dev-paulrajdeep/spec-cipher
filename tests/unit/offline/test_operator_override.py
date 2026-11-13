from __future__ import annotations

import pytest

from speccipher.domain.errors import OverrideValidationError
from speccipher.ingestion.overrides.loader import validate_operator_override


def _override() -> dict[str, object]:
    return {
        "source_filename": "SP 7_2026-401-800.pdf",
        "operator": "release-operator",
        "timestamp": "2026-11-12T09:30:00Z",
        "rationale": "Title page is not repeated in this volume.",
        "asserted_standard": {
            "family": "SP",
            "number": "7",
            "year": 2026,
            "code": "SP 7:2026",
        },
    }


def test_override_preserves_audited_identity() -> None:
    override = validate_operator_override(_override())

    assert override.asserted_standard.code == "SP 7:2026"
    assert override.operator == "release-operator"
    assert override.timestamp.endswith("Z")
    assert override.evidence_sources == ()


def test_version_two_override_preserves_required_audit_evidence() -> None:
    override = validate_operator_override(
        {
            "asserted_identity": "SP 7:2026",
            "operator_id": "operator-17",
            "timestamp": "2026-11-12T09:30:00Z",
            "reason": "Volume title header is absent; source sequence confirms the work.",
            "evidence_sources": ["officialBIS/volume-2.pdf", "BIS volume index"],
        }
    )

    assert override.asserted_identity.code == "SP 7:2026"
    assert override.operator_id == "operator-17"
    assert override.to_dict()["evidence_sources"] == [
        "officialBIS/volume-2.pdf",
        "BIS volume index",
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("asserted_standard", None, "asserted_standard"),
        ("timestamp", "not-a-date", "ISO-8601"),
        ("timestamp", "2026-11-12T09:30:00", "timezone"),
        ("operator", "   ", "operator"),
        ("asserted_standard", {"family": "SP"}, "Invalid asserted_standard"),
        (
            "asserted_standard",
            {"family": "SP", "number": "7", "year": 2026, "code": "IS 456:2000"},
            "does not match",
        ),
    ],
)
def test_override_rejects_missing_or_invalid_audit_data(
    field: str, value: object, message: str
) -> None:
    payload = _override()
    payload[field] = value

    with pytest.raises(OverrideValidationError, match=message):
        validate_operator_override(payload)


def test_version_two_override_requires_evidence_sources() -> None:
    payload = {
        "asserted_identity": "SP 7:2026",
        "operator_id": "operator-17",
        "timestamp": "2026-11-12T09:30:00Z",
        "reason": "Volume title header is absent.",
        "evidence_sources": [],
    }

    with pytest.raises(OverrideValidationError, match="evidence_sources must not be empty"):
        validate_operator_override(payload)
