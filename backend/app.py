"""SpecCipher's small, same-origin Flask serving surface."""
from __future__ import annotations
import atexit, logging, os, sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from flask import Flask, g, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, BadRequest, RequestEntityTooLarge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.preflight import PreflightError, validate_bundle
from backend.qwen_reasoning import QwenReasoner, deterministic_explanation
from intelligence.pipeline import IntelligenceEngine
from intelligence.relationships import CorpusKnowledgeBase
from retrieval.config import RetrievalConfig
from retrieval.corpus_config import (
    resolve_knowledge_base_paths,
    resolve_retrieval_corpus_path,
    serving_corpus_origin,
)
from retrieval.indexing import IndexCompatibilityError
from retrieval.service import RetrievalService, RetrievalUnavailableError

MAX_INPUT_CHARACTERS, RETRIEVAL_CANDIDATE_K, FINAL_TOP_K = 4_000, 500, 5
logger = logging.getLogger("speccipher")

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("SPEC_CIPHER_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
FRONTEND_DIR, RETRIEVAL_INDEX_DIR = PROJECT_ROOT / "frontend", PROJECT_ROOT / "retrieval-index"
CANONICAL_CORPUS_FILE = resolve_retrieval_corpus_path()
hf_offline = _env_flag("SPEC_CIPHER_HF_OFFLINE")
if hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _load_runtime() -> tuple[RetrievalService, IntelligenceEngine, QwenReasoner]:
    """Validate static inputs before loading the one process-wide embedding model."""
    started = perf_counter()
    knowledge_paths = resolve_knowledge_base_paths()
    try:
        validate_bundle(RETRIEVAL_INDEX_DIR, CANONICAL_CORPUS_FILE)
        service = RetrievalService.from_persisted(RETRIEVAL_INDEX_DIR, config=RetrievalConfig(
            embedding_device=os.getenv("SPEC_CIPHER_EMBEDDING_DEVICE", "auto").strip().lower(),
            allow_cpu_fallback=_env_flag("SPEC_CIPHER_ALLOW_CPU_FALLBACK", True),
            reranker_enabled=_env_flag("SPEC_CIPHER_RERANKER_ENABLED"), sequential_model_lifecycle=False,
            corpus_path=CANONICAL_CORPUS_FILE,
        ))
    except (PreflightError, FileNotFoundError, RuntimeError, ValueError, IndexCompatibilityError) as exc:
        logger.error(
            "Startup validation failed type=%s detail=%s",
            exc.__class__.__name__,
            str(exc)[:300],
        )
        raise RuntimeError(
            "SpecCipher startup failed. Verify the matching retrieval-index bundle, "
            "canonical corpus, cached BAAI/bge-m3 artifacts, and embedding device configuration."
        ) from exc
    if not knowledge_paths:
        raise RuntimeError(
            "SpecCipher startup failed. No canonical corpus documents were found under corpus/data."
        )
    knowledge_base = CorpusKnowledgeBase.from_canonical_json(knowledge_paths)
    readiness = service.readiness()
    logger.info(
        "Runtime ready offline=%s chunks=%s standards=%s device=%s ollama_model=%s startup_ms=%.2f",
        hf_offline,
        readiness.get("chunk_count"),
        readiness.get("standard_count"),
        readiness.get("embedding_device"),
        QwenReasoner().config.model_name,
        (perf_counter() - started) * 1000,
    )
    return service, IntelligenceEngine(knowledge_base=knowledge_base), QwenReasoner()

retriever, intelligence, qwen_reasoner = _load_runtime()
close_retriever = getattr(retriever, "close", None)
if callable(close_retriever):
    atexit.register(close_retriever)

def _error(code: str, message: str, status: int):
    return jsonify({"status": "error", "result_state": code, "message": message}), status

def _cors_origins() -> set[str]:
    return {item.strip() for item in os.getenv("SPEC_CIPHER_CORS_ORIGINS", "").split(",") if item.strip()}

@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    origin = request.headers.get("Origin")
    allowed = _cors_origins()
    if origin and origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Max-Age"] = "600"
        response.headers["Vary"] = "Origin"
    started = getattr(g, "request_started", None)
    if started is not None:
        logger.info(
            "HTTP request method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            request.path,
            response.status_code,
            (perf_counter() - started) * 1000,
        )
    return response


@app.before_request
def _start_request_timer():
    g.request_started = perf_counter()

@app.errorhandler(RequestEntityTooLarge)
def _too_large(_value): return _error("invalid_request", "Request body is too large.", 413)
@app.errorhandler(BadRequest)
def _bad_request(_value): return _error("invalid_request", "Request body must be valid JSON.", 400)


@app.errorhandler(HTTPException)
def _http_error(error: HTTPException):
    """Keep routing and method failures on the same non-sensitive JSON contract."""
    status = error.code or 500
    if status >= 500:
        result_state = "internal_error"
        message = "The analysis service encountered an internal error."
    else:
        result_state = "invalid_request"
        messages = {
            404: "The requested endpoint was not found.",
            405: "The requested method is not supported.",
            415: "Content-Type must be application/json.",
        }
        message = messages.get(status, "The request could not be processed.")
    if status >= 500:
        logger.error("HTTP failure status=%d type=%s", status, error.__class__.__name__)
    return _error(result_state, message, status)


@app.errorhandler(Exception)
def _unexpected_error(error: Exception):
    """Prevent framework-generated HTML or exception details from reaching clients."""
    logger.exception("Unhandled HTTP failure type=%s", error.__class__.__name__)
    return _error("internal_error", "The analysis service encountered an internal error.", 500)

@app.route("/")
def index(): return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/health")
def health():
    state = retriever.readiness()
    status = "ready" if state["ready"] and state["model_loaded"] else "not_ready"
    payload = {
        "status": status,
        "model_loaded": state["model_loaded"],
        "embedding_model": state["embedding_model"],
        "embedding_device": state["embedding_device"],
        "hf_offline": hf_offline,
        "corpus": CANONICAL_CORPUS_FILE.name,
        "corpus_origin": serving_corpus_origin(),
        "chunk_count": state.get("chunk_count", 0),
        "standard_count": state.get("standard_count", 0),
        "ollama_model": qwen_reasoner.config.model_name,
        "explanation_optional": True,
        "limitations": (
            "Historical demo checklist is used when no authentic officialBIS-derived "
            "corpus is present. Generative explanation is optional."
            if serving_corpus_origin() == "historical_demo_checklist"
            else "Recommendations are deterministic; generative explanation is optional."
        ),
    }
    return jsonify(payload), 200 if status == "ready" else 503

@app.route("/get_result", methods=["GET", "POST", "OPTIONS"])
def analyse_input():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.method != "POST":
        return _error("invalid_request", "The requested method is not supported.", 405)
    started = perf_counter()
    readiness = retriever.readiness()
    if not (readiness.get("ready") and readiness.get("model_loaded")):
        logger.warning("Analysis rejected because runtime is not_ready")
        return _error("not_ready", "SpecCipher is not ready to analyse requirements yet.", 503)
    if not request.is_json: return _error("invalid_request", "Content-Type must be application/json.", 415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict): return _error("invalid_request", "Request body must be a JSON object.", 400)
    user_input = data.get("user_input")
    if not isinstance(user_input, str): return _error("invalid_request", "user_input must be a string.", 400)
    user_input = user_input.strip()
    if not user_input: return _error("invalid_request", "user_input is required.", 400)
    if len(user_input) > MAX_INPUT_CHARACTERS: return _error("invalid_request", f"user_input must be at most {MAX_INPUT_CHARACTERS} characters.", 400)
    try:
        results = retriever.search(user_input, top_k=RETRIEVAL_CANDIDATE_K)
        run = intelligence.recommend(user_input, results)
        recommendations = _serialize_recommendations(user_input, run.recommendations[:FINAL_TOP_K])
    except (RetrievalUnavailableError, FileNotFoundError, IndexCompatibilityError):
        logger.warning("Retrieval dependency unavailable")
        return _error("dependency_unavailable", "Retrieval dependency is unavailable.", 503)
    except ValueError:
        logger.warning("Invalid value reached analysis pipeline")
        return _error("invalid_request", "The requirement could not be processed.", 400)
    except RuntimeError:
        logger.exception("Analysis dependency failed")
        return _error("dependency_unavailable", "Analysis dependency is unavailable.", 503)
    except Exception:
        logger.exception("Unexpected analysis failure")
        return _error("internal_error", "The analysis service encountered an internal error.", 500)
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
        "Analysis completed result_state=%s candidates=%d reranked=%d rejected=%d recommendations=%d duration_ms=%.2f",
        result_state, len(results), reranked_count, rejected_count, len(recommendations),
        (perf_counter() - started) * 1000,
    )
    return jsonify({
        "status": "success",
        "result_state": result_state,
        "user_input": user_input,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieved_candidate_count": len(results),
        "reranked_candidate_count": reranked_count,
        "rejected_candidate_count": rejected_count,
        "highlight_terms": _highlight_terms(user_input),
        "recommendations": recommendations,
        "warnings": warnings,
    })

def _serialize_recommendations(user_input, recommendations):
    payloads = []
    for index, recommendation in enumerate(recommendations):
        payload = recommendation.to_dict()
        if index == 0:
            outcome = qwen_reasoner.explain(user_input, recommendation)
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
        else:
            explanation = deterministic_explanation(recommendation)
            payload["explanation_mode"] = "deterministic"
            payload["explanation_label"] = "Evidence-grounded summary"
        payload["explanation"], payload["explanation_text"] = explanation, explanation["summary"]
        payloads.append(payload)
    return payloads

def _highlight_terms(user_input):
    stop_words = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return list(dict.fromkeys(word.strip(".,:;()[]{}") for word in user_input.split() if len(word.strip(".,:;()[]{}")) > 2 and word.strip(".,:;()[]{}").lower() not in stop_words))

@app.route("/<path:filename>")
def frontend_files(filename): return send_from_directory(FRONTEND_DIR, filename)

if __name__ == "__main__":
    app.run(host=os.getenv("SPEC_CIPHER_HOST", "127.0.0.1"), port=int(os.getenv("SPEC_CIPHER_PORT", "5000")), debug=False, use_reloader=False)
