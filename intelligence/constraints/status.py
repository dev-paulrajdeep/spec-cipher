"""Status-based hard exclusions using corpus metadata only."""

from __future__ import annotations

from corpus.models import StandardMetadata
from intelligence.models import ConstraintReason


_UNUSABLE_STATUSES = {"withdrawn", "superseded", "cancelled"}


def status_reasons(metadata: StandardMetadata | None) -> tuple[ConstraintReason, ...]:
    if metadata and metadata.status and metadata.status.lower() in _UNUSABLE_STATUSES:
        return (ConstraintReason("unusable_status", f"Standard status is {metadata.status.lower()}."),)
    return ()
