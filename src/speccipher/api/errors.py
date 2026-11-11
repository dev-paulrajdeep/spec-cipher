"""Error handling supporting both exact legacy contracts and RFC 7807 Problem Details."""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

logger = logging.getLogger("speccipher.errors")


def register_error_handlers(app: Flask) -> None:
    """Register sanitized error responses preserving test and client contracts."""

    def _format_error(
        result_state: str, message: str, status: int, title: str | None = None
    ) -> Any:
        accept_header = request.headers.get("Accept", "")
        is_v1 = request.path.startswith("/v1/")
        is_problem = "application/problem+json" in accept_header

        if is_v1 or is_problem:
            # RFC 7807 Problem Details response
            payload = {
                "type": "about:blank",
                "title": title or ("Invalid Request" if status < 500 else "Internal Server Error"),
                "status": status,
                "detail": message,
                "instance": request.path,
                "result_state": result_state,
                "message": message,
            }
            response = jsonify(payload)
            response.content_type = "application/problem+json"
            return response, status

        # Legacy contract: Exact 3-key format expected by tests
        return jsonify(
            {
                "status": "error",
                "result_state": result_state,
                "message": message,
            }
        ), status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_value: RequestEntityTooLarge) -> Any:
        return _format_error(
            "invalid_request", "Request body is too large.", 413, "Payload Too Large"
        )

    @app.errorhandler(BadRequest)
    def _bad_request(_value: BadRequest) -> Any:
        return _format_error(
            "invalid_request", "Request body must be valid JSON.", 400, "Bad Request"
        )

    @app.errorhandler(HTTPException)
    def _http_error(error: HTTPException) -> Any:
        status = error.code or 500
        if status >= 500:
            result_state = "internal_error"
            message = "The analysis service encountered an internal error."
            title = "Internal Server Error"
            logger.error("HTTP failure status=%d type=%s", status, error.__class__.__name__)
        else:
            result_state = "invalid_request"
            title = "Invalid Request"
            messages = {
                404: "The requested endpoint was not found.",
                405: "The requested method is not supported.",
                415: "Content-Type must be application/json.",
            }
            message = messages.get(status, "The request could not be processed.")
        return _format_error(result_state, message, status, title)

    @app.errorhandler(Exception)
    def _unexpected_error(error: Exception) -> Any:
        logger.exception("Unhandled HTTP failure type=%s", error.__class__.__name__)
        return _format_error(
            "internal_error",
            "The analysis service encountered an internal error.",
            500,
            "Internal Server Error",
        )
