"""SpecCipher's small, same-origin Flask serving surface."""
from __future__ import annotations
import atexit, logging, os, sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from backend.preflight import PreflightError, validate_bundle
from backend.qwen_reasoning import EXPLANATION_FALLBACK_WARNING, QwenReasoner, deterministic_explanation
from intelligence.pipeline import IntelligenceEngine
from intelligence.relationships import CorpusKnowledgeBase
from retrieval.config import RetrievalConfig
from retrieval.corpus_config import CANONICAL_CORPUS_PATH
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
CANONICAL_CORPUS_FILE = CANONICAL_CORPUS_PATH
hf_offline = _env_flag("SPEC_CIPHER_HF_OFFLINE")
if hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _load_runtime() -> tuple[RetrievalService, IntelligenceEngine, QwenReasoner]:
    """Validate static inputs before loading the one process-wide embedding model."""
    started = perf_counter()
    try:
        validate_bundle(RETRIEVAL_INDEX_DIR, CANONICAL_CORPUS_FILE)
        service = RetrievalService.from_persisted(RETRIEVAL_INDEX_DIR, config=RetrievalConfig(
            embedding_device=os.getenv("SPEC_CIPHER_EMBEDDING_DEVICE", "auto").strip().lower(),
            allow_cpu_fallback=_env_flag("SPEC_CIPHER_ALLOW_CPU_FALLBACK", True),
            reranker_enabled=_env_flag("SPEC_CIPHER_RERANKER_ENABLED"), sequential_model_lifecycle=False,
            corpus_path=CANONICAL_CORPUS_FILE,
        ))
    except (PreflightError, FileNotFoundError, RuntimeError, ValueError, IndexCompatibilityError) as exc:
        logger.error("Startup validation failed: %s", exc.__class__.__name__)
        raise RuntimeError("SpecCipher startup failed. Verify the matching retrieval-index bundle, canonical corpus, cached BAAI/bge-m3 artifacts, and embedding device configuration.") from exc
    knowledge_base = CorpusKnowledgeBase.from_canonical_json((CANONICAL_CORPUS_FILE,))
    logger.info("Runtime ready offline=%s startup_ms=%.2f", hf_offline, (perf_counter() - started) * 1000)
    return service, IntelligenceEngine(knowledge_base=knowledge_base), QwenReasoner()

retriever, intelligence, qwen_reasoner = _load_runtime()
close_retriever = getattr(retriever, "close", None)
if callable(close_retriever):
    atexit.register(close_retriever)

def _error(code: str, message: str, status: int):
    return jsonify({"status": "error", "result_state": code, "message": message}), status

@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    origin = request.headers.get("Origin")
    allowed = {item.strip() for item in os.getenv("SPEC_CIPHER_CORS_ORIGINS", "").split(",") if item.strip()}
    if origin and origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response

@app.errorhandler(RequestEntityTooLarge)
def _too_large(_value): return _error("invalid_request", "Request body is too large.", 413)
@app.errorhandler(BadRequest)
def _bad_request(_value): return _error("invalid_request", "Request body must be valid JSON.", 400)

@app.route("/")
def index(): return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/health")
def health():
    state = retriever.readiness()
    status = "ready" if state["ready"] and state["model_loaded"] else "not_ready"
    return jsonify({"status": status, "model_loaded": state["model_loaded"], "embedding_model": state["embedding_model"], "embedding_device": state["embedding_device"], "hf_offline": hf_offline, "corpus": CANONICAL_CORPUS_FILE.name, "limitations": "One independently indexed standard and one chunk are currently available."}), 200 if status == "ready" else 503

@app.route("/get_result", methods=["POST"])
def analyse_input():
    started = perf_counter()
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
    warnings = []
    if not recommendations:
        warnings.append("No sufficiently supported standard found for this requirement.")
        if results: warnings.append(f"{len(results)} candidate retrieved but rejected by applicability/evidence thresholds.")
    logger.info("Analysis completed candidates=%d reranked=%d recommendations=%d duration_ms=%.2f", len(results), reranked_count, len(recommendations), (perf_counter() - started) * 1000)
    return jsonify({"status": "success", "result_state": "recommendations" if recommendations else "no_match", "user_input": user_input, "analysis_timestamp": datetime.now(timezone.utc).isoformat(), "retrieved_candidate_count": len(results), "reranked_candidate_count": reranked_count, "rejected_candidate_count": rejected_count, "highlight_terms": _highlight_terms(user_input), "recommendations": recommendations, "warnings": warnings})

def _serialize_recommendations(user_input, recommendations):
    payloads = []
    for recommendation in recommendations:
        payload = recommendation.to_dict()
        outcome = qwen_reasoner.explain(user_input, recommendation)
        explanation = outcome.explanation or deterministic_explanation(recommendation)
        payload["explanation"], payload["explanation_text"] = explanation, explanation["summary"]
        if outcome.warning: payload["warnings"].append(EXPLANATION_FALLBACK_WARNING)
        payloads.append(payload)
    return payloads

def _highlight_terms(user_input):
    stop_words = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return list(dict.fromkeys(word.strip(".,:;()[]{}") for word in user_input.split() if len(word.strip(".,:;()[]{}")) > 2 and word.strip(".,:;()[]{}").lower() not in stop_words))

@app.route("/<path:filename>")
def frontend_files(filename): return send_from_directory(FRONTEND_DIR, filename)

if __name__ == "__main__":
    app.run(host=os.getenv("SPEC_CIPHER_HOST", "127.0.0.1"), port=int(os.getenv("SPEC_CIPHER_PORT", "5000")), debug=False, use_reloader=False)
