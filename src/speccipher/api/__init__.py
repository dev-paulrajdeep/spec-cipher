"""Presentation & Delivery Layer (Layer 1).

Exposes the decoupled Flask application factory, HTTP middleware,
RFC 7807 Problem Details error handling, and API blueprints.
"""

from __future__ import annotations

from speccipher.api.app import create_app

__all__ = ["create_app"]
