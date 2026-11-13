"""Offline publication assembly and canonical-record projection."""

from .records import CanonicalRecordError, records_from_canonical_documents

__all__ = ["CanonicalRecordError", "records_from_canonical_documents"]
