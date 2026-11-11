"""SpecCipher Pure Domain Exceptions (Layer 7).

Constraint: Python standard library ONLY (zero third-party imports).
"""

from __future__ import annotations


class DomainError(Exception):
    """Base exception for all SpecCipher domain invariant violations."""


class MultipartAssemblyError(DomainError):
    """Raised when multi-volume publication assembly invariants are violated."""


class OverrideValidationError(DomainError):
    """Raised when an operator override is malformed, missing audit fields, or invalid."""


class ProvenanceValidationError(DomainError):
    """Raised when document chunks or sections have out-of-range page numbers or inconsistent provenance."""
