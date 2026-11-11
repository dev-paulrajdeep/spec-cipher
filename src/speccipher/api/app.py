"""Decoupled Flask application factory for SpecCipher."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from flask import Flask, current_app, jsonify, request, send_from_directory

from speccipher.api.errors import register_error_handlers
from speccipher.api.middleware import register_middleware
from speccipher.services.container import ApplicationContainer

logger = logging.getLogger("speccipher.api")


def _serialize_recommendations(
    user_input: str,
    recommendations: tuple[Any, ...],
    reasoner: Any,
) -> list[dict[str, Any]]:
    """Serialize recommendations attaching fail-open explanations without mutating scores."""
    from backend.qwen_reasoning import deterministic_explanation

    payloads = []
    for index, recommendation in enumerate(recommendations):
        before_qwen = recommendation.to_dict()
        payload = dict(before_qwen)

        if index == 0 and reasoner is not None:
            outcome = reasoner.explain(user_input, recommendation)
            explanation = outcome.explanation or deterministic_explanation(recommendation)
            mode = outcome.mode or ("deterministic" if outcome.warning else "generative")
            payload["explanation_mode"] = mode
            payload["explanation_label"] = (
                "Generative evidence-grounded summary"
                if mode == "generative"
                else "Evidence-grounded summary"
            )
            if outcome.warning:
                payload["explanation_note"] = outcome.warning
            logger.info(
                "Explanation attached mode=%s fallback=%s standard=%s",
                mode,
                outcome.fallback_reason or "none",
                before_qwen.get("standard_number"),
            )
        else:
            explanation = deterministic_explanation(recommendation)
            payload["explanation_mode"] = "deterministic"
            payload["explanation_label"] = "Evidence-grounded summary"

        payload["explanation"] = explanation
        payload["explanation_text"] = explanation["summary"]

        # Ensure Qwen never mutates ranking, applicability, or scores
        for key, value in before_qwen.items():
            if key not in {"explanation", "explanation_text", "warnings"}:
                payload[key] = value

        payloads.append(payload)
    return payloads


def _highlight_terms(user_input: str) -> list[str]:
    stop_words = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return list(
        dict.fromkeys(
            word.strip(".,:;()[]{}")
            for word in user_input.split()
            if len(word.strip(".,:;()[]{}")) > 2
            and word.strip(".,:;()[]{}").lower() not in stop_words
        )
    )


def create_app(
    container: ApplicationContainer | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> Flask:
    """Construct and configure a decoupled Flask application instance.

    Guarantees:
    - Zero heavy models loaded during create_app execution.
    - DI container attached to app.config["CONTAINER"].
    - Strict adherence to public HTTP contracts and security headers.
    """
    app = Flask(__name__)
    resolved_container = container or ApplicationContainer()
    app.config["CONTAINER"] = resolved_container
    app.config["MAX_CONTENT_LENGTH"] = resolved_container.settings.server.max_content_length

    if config_overrides:
        app.config.update(config_overrides)

    register_middleware(app, resolved_container)
    register_error_handlers(app)

    frontend_dir = resolved_container.settings.paths.frontend_dir

    # Static routes
    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:filename>")
    def frontend_files(filename: str):
        return send_from_directory(frontend_dir, filename)

    # Health endpoints
    @app.route("/health/live")
    def health_live():
        return jsonify({"status": "live", "timestamp": datetime.now(UTC).isoformat()}), 200

    @app.route("/health/ready")
    def health_ready():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        state = c.readiness()
        return jsonify(state), 200 if state.get("ready") else 503

    @app.route("/health")
    def legacy_health():
        c: ApplicationContainer = current_app.config["CONTAINER"]
        import sys

        app_module = sys.modules.get("backend.app")
        retriever = (
            app_module.__dict__["retriever"]
            if app_module and "retriever" in app_module.__dict__
            else c.retrieval_service
        )
        reasoner = (
            app_module.__dict__["qwen_reasoner"]
            if app_module and "qwen_reasoner" in app_module.__dict__
            else c.explanation_service
        )

        readiness_state = getattr(
            retriever, "readiness", lambda: c.readiness().get("retrieval", {})
        )()
        is_ready = bool(readiness_state.get("ready") and readiness_state.get("model_loaded"))
        status = "ready" if is_ready else "not_ready"
        from retrieval.corpus_config import serving_corpus_origin

        ollama_model = getattr(
            getattr(reasoner, "config", None), "model_name", c.settings.explanation.model_name
        )
        payload = {
            "status": status,
            "model_loaded": readiness_state.get("model_loaded", False),
            "embedding_model": readiness_state.get(
                "embedding_model", c.settings.retrieval.embedding_model
            ),
            "embedding_device": readiness_state.get(
                "embedding_device", c.settings.retrieval.embedding_device
            ),
            "hf_offline": c.settings.retrieval.hf_offline,
            "corpus": "corpus.chunks.jsonl",
            "corpus_origin": serving_corpus_origin(),
            "chunk_count": readiness_state.get("chunk_count", 0),
            "standard_count": readiness_state.get("standard_count", 0),
            "ollama_model": ollama_model,
            "explanation_optional": True,
            "limitations": (
                "Historical demo checklist is used when no authentic officialBIS-derived "
                "corpus is present. Generative explanation is optional."
                if serving_corpus_origin() == "historical_demo_checklist"
                else "Recommendations are deterministic; generative explanation is optional."
            ),
        }
        return jsonify(payload), 200 if status == "ready" else 503

    # Legacy analysis endpoint
    @app.route("/get_result", methods=["GET", "POST", "OPTIONS"])
    def get_result():
        if request.method == "OPTIONS":
            return ("", 204)
        if request.method != "POST":
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "The requested method is not supported.",
                }
            ), 405

        c: ApplicationContainer = current_app.config["CONTAINER"]
        started = perf_counter()

        import sys

        app_module = sys.modules.get("backend.app")
        retriever = (
            app_module.__dict__["retriever"]
            if app_module and "retriever" in app_module.__dict__
            else c.retrieval_service
        )
        intelligence = (
            app_module.__dict__["intelligence"]
            if app_module and "intelligence" in app_module.__dict__
            else c.decision_engine
        )
        reasoner = (
            app_module.__dict__["qwen_reasoner"]
            if app_module and "qwen_reasoner" in app_module.__dict__
            else c.explanation_service
        )

        readiness = getattr(retriever, "readiness", lambda: {"ready": True, "model_loaded": True})()
        if not (readiness.get("ready") and readiness.get("model_loaded")):
            logger.warning("Analysis rejected: runtime not ready")
            return jsonify(
                {
                    "status": "error",
                    "result_state": "not_ready",
                    "message": "SpecCipher is not ready to analyse requirements yet.",
                }
            ), 503

        if not request.is_json:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "Content-Type must be application/json.",
                }
            ), 415

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "Request body must be a JSON object.",
                }
            ), 400

        user_input = data.get("user_input")
        if not isinstance(user_input, str):
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "user_input must be a string.",
                }
            ), 400

        user_input = user_input.strip()
        if not user_input:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": "user_input is required.",
                }
            ), 400

        max_chars = c.settings.server.max_input_characters
        if len(user_input) > max_chars:
            return jsonify(
                {
                    "status": "error",
                    "result_state": "invalid_request",
                    "message": f"user_input must be at most {max_chars} characters.",
                }
            ), 400

        try:
            results = retriever.search(user_input, top_k=c.settings.retrieval.top_k)
            run = intelligence.recommend(user_input, results)

            recommendations = _serialize_recommendations(
                user_input,
                run.recommendations[: c.settings.retrieval.final_top_k],
                reasoner,
            )
        except Exception as exc:
            from retrieval.indexing import IndexCompatibilityError
            from retrieval.service import RetrievalUnavailableError

            if isinstance(
                exc, (RetrievalUnavailableError, FileNotFoundError, IndexCompatibilityError)
            ):
                logger.warning("Retrieval dependency unavailable: %s", exc)
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "dependency_unavailable",
                        "message": "Retrieval dependency is unavailable.",
                    }
                ), 503
            if isinstance(exc, ValueError):
                logger.warning("Invalid value reached pipeline: %s", exc)
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "invalid_request",
                        "message": "The requirement could not be processed.",
                    }
                ), 400
            if isinstance(exc, RuntimeError):
                logger.exception("Pipeline runtime dependency failed: %s", exc)
                return jsonify(
                    {
                        "status": "error",
                        "result_state": "dependency_unavailable",
                        "message": "Analysis dependency is unavailable.",
                    }
                ), 503
            logger.exception("Unexpected analysis failure: %s", exc)
            return jsonify(
                {
                    "status": "error",
                    "result_state": "internal_error",
                    "message": "The analysis service encountered an internal error.",
                }
            ), 500

        diagnostics = tuple(getattr(run, "candidates", ()))
        reranked_count = sum(item.rerank_result is not None for item in diagnostics)
        rejected_count = max(0, len(diagnostics) - len(run.recommendations))
        result_state = "recommendations" if recommendations else "no_match"
        warnings = []
        if not recommendations:
            warnings.append("No sufficiently supported standard found for this requirement.")
            if results:
                label = "candidate" if len(results) == 1 else "candidates"
                warnings.append(
                    f"{len(results)} {label} retrieved but rejected by applicability/evidence thresholds."
                )

        logger.info(
            "Analysis complete result_state=%s candidates=%d reranked=%d rejected=%d recs=%d duration_ms=%.2f",
            result_state,
            len(results),
            reranked_count,
            rejected_count,
            len(recommendations),
            (perf_counter() - started) * 1000,
        )

        return jsonify(
            {
                "status": "success",
                "result_state": result_state,
                "user_input": user_input,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
                "retrieved_candidate_count": len(results),
                "reranked_candidate_count": reranked_count,
                "rejected_candidate_count": rejected_count,
                "highlight_terms": _highlight_terms(user_input),
                "recommendations": recommendations,
                "warnings": warnings,
            }
        )

    return app
