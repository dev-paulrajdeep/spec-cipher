"""HTTP request tracing, timing, security headers, and CORS middleware."""

from __future__ import annotations

import logging
import uuid
from time import perf_counter

from flask import Flask, Response, g, request

from speccipher.services.container import ApplicationContainer

logger = logging.getLogger("speccipher.access")


def register_middleware(app: Flask, container: ApplicationContainer) -> None:
    """Attach request ID, timing, security headers, and CORS handlers to Flask."""

    @app.before_request
    def _before_request() -> None:
        # Trace ID generation/propagation
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        g.request_started = perf_counter()

    @app.after_request
    def _after_request(response: Response) -> Response:
        # Standard security headers
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )

        # CORS handling
        origin = request.headers.get("Origin")
        allowed = set(container.settings.server.cors_origins)
        if origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, X-Request-ID"
            response.headers["Access-Control-Max-Age"] = "600"
            response.headers["Vary"] = "Origin"

        # Structured request logging
        started = getattr(g, "request_started", None)
        if started is not None:
            duration_ms = (perf_counter() - started) * 1000
            logger.info(
                "HTTP request method=%s path=%s status=%d duration_ms=%.2f request_id=%s",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
                getattr(g, "request_id", "none"),
            )
        return response
