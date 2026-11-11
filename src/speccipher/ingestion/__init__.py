"""Ingestion Plane Engine Layer (Layer 4 Sibling).

Offline build plane engine for PDF parsing, multi-part document assembly,
absolute page numbering, operator identity overrides, and chunk generation.

INVARIANT: Serving plane modules must NEVER import from speccipher.ingestion!
"""

from __future__ import annotations

__all__: list[str] = []
