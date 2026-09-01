"""Select and format direct retrieved-chunk evidence; no second retrieval system."""

from .formatter import format_evidence
from .locator import EvidenceLocator

__all__ = ["EvidenceLocator", "format_evidence"]
