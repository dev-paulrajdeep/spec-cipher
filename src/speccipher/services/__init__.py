"""Application Services & Composition Root Layer (Layer 2).

Provides the unified ApplicationContainer for dependency injection,
lifecycle management, and application use-case coordinators.
"""

from __future__ import annotations

from speccipher.services.container import ApplicationContainer

__all__ = ["ApplicationContainer"]
