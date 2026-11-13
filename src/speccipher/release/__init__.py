"""Corpus Release & Lifecycle Management Layer (Layer 3).

Mediates the contract between the offline build plane and online serving plane.
Manages immutable versioned Corpus Releases, manifest schemas, checksum verification,
and atomic publication via CURRENT pointer.
"""

from __future__ import annotations

from .manifest import (
    ReleaseIntegrityError,
    ReleaseNotFoundError,
    resolve_release_path,
    verify_release,
)
from .publisher import ReleasePublisher, ReleasePublishError, publish_release

__all__ = [
    "ReleaseIntegrityError",
    "ReleaseNotFoundError",
    "ReleasePublishError",
    "ReleasePublisher",
    "publish_release",
    "resolve_release_path",
    "verify_release",
]
