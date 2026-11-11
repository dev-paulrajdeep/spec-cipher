"""Corpus Release & Lifecycle Management Layer (Layer 3).

Mediates the contract between the offline build plane and online serving plane.
Manages immutable versioned Corpus Releases, manifest schemas, checksum verification,
and atomic publication via CURRENT pointer.
"""

from __future__ import annotations

__all__: list[str] = []
