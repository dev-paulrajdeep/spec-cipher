from flask import Flask, send_from_directory, request, jsonify
import atexit
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
# CHANGING ROOT PATH THAT PYTHON USES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.service import RetrievalService, RetrievalUnavailableError
from retrieval.indexing import IndexCompatibilityError 
from retrieval.config import RetrievalConfig
from retrieval.corpus_config import CANONICAL_CORPUS_PATH
from intelligence.pipeline import IntelligenceEngine
from intelligence.relationships import CorpusKnowledgeBase
from backend.qwen_reasoning import QwenReasoner
from backend.preflight import PreflightError, validate_bundle

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Frontend folder is beside the backend folder
FRONTEND_DIR = os.path.join(app.root_path, "../frontend")

# Generated retrieval bundle; build it before starting Flask.
RETRIEVAL_INDEX_DIR = Path(app.root_path).parent / "retrieval-index"
CANONICAL_CORPUS_FILE = CANONICAL_CORPUS_PATH
embedding_device = os.getenv("SPEC_CIPHER_EMBEDDING_DEVICE", "auto").strip().lower()
allow_cpu_fallback = os.getenv("SPEC_CIPHER_ALLOW_CPU_FALLBACK", "1") == "1"
reranker_enabled = os.getenv("SPEC_CIPHER_RERANKER_ENABLED", "0") == "1"
logging.info(
    "Loading retrieval service: device=%s allow_cpu_fallback=%s reranker_enabled=%s",
    embedding_device,
    allow_cpu_fallback,
    reranker_enabled,
)
try:
    validate_bundle(
        RETRIEVAL_INDEX_DIR,
        CANONICAL_CORPUS_FILE,
        check_ollama=os.getenv("SPEC_CIPHER_PREFLIGHT_OLLAMA", "0") == "1",
        ollama_host=os.getenv("QWEN_OLLAMA_HOST", "http://localhost:11434"),
    )
    retriever = RetrievalService.from_persisted(
        RETRIEVAL_INDEX_DIR,
        config=RetrievalConfig(
            embedding_device=embedding_device,
            allow_cpu_fallback=allow_cpu_fallback,
            reranker_enabled=reranker_enabled,
            sequential_model_lifecycle=False,
            corpus_path=CANONICAL_CORPUS_FILE,
        ),
    )
except (PreflightError, FileNotFoundError, RuntimeError) as error:
    raise RuntimeError(
        "SpecCipher startup preflight failed. Check the retrieval bundle, configured corpus, "
        "embedding model, and CPU/GPU settings."
    ) from error
close_retriever = getattr(retriever, "close", None)
if callable(close_retriever):
    atexit.register(close_retriever)

knowledge_base = (
    CorpusKnowledgeBase.from_canonical_json((CANONICAL_CORPUS_FILE,))
    if CANONICAL_CORPUS_FILE.exists()
    else CorpusKnowledgeBase()
)
intelligence = IntelligenceEngine(knowledge_base=knowledge_base)
qwen_reasoner = QwenReasoner()

RETRIEVAL_CANDIDATE_K = 500
FINAL_TOP_K = 5



# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# --------------------------------------------------
# Search / Recommendation API
# --------------------------------------------------

@app.route("/get_result", methods=["POST"])
def analyse_input():

    # Get JSON sent by frontend
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON data received."
        }), 400

    # Get user's input
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({
            "status": "error",
            "message": "user_input is required."
        }), 400

    logging.debug(f"user_input: {user_input}")

    try:
        # Flask's development server is threaded. Reuse RetrievalService's
        # re-entrant lifecycle lock so another request cannot load BGE models
        # while Ollama is generating an explanation for this request.
        with retriever._search_lock:
            results = retriever.search(user_input, top_k=RETRIEVAL_CANDIDATE_K)
            run = intelligence.recommend(user_input, results)
            recommendations = _serialize_recommendations(
                user_input, run.recommendations[:FINAL_TOP_K]
            )

    # handling the 4 errors specifically asked     
    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400

    except FileNotFoundError as error:
        logging.exception("File not found")

        return jsonify({
        "status": "error",
        "message": "Retrieval service unavailable."
        }), 503

    except RetrievalUnavailableError as error:
        logging.exception("Retrieval service unavailable")
        return jsonify({
        "status": "error",
        "result_state": "unavailable",
        "message": "Retrieval service unavailable."
        }), 503

    except IndexCompatibilityError as error:
        logging.exception("Retrieval index is incompatible")
        return jsonify({
        "status": "error",
        "result_state": "unavailable",
        "message": "Retrieval service unavailable."
        }), 503

    except Exception as error:
        logging.exception("Recommendation pipeline failed")
        return jsonify({
            "status": "error",
            "result_state": "unavailable",
            "message": "Recommendation service unavailable.",
        }), 503

    # getting rid of all exception as said 4 errors are handled above
    # except Exception as error:
    #     logging.error(f"Retrieval error: {error}")

    #     return jsonify({
    #         "status": "error",
    #         "message": "Retrieval service unavailable."
    #     }), 500

    return jsonify({
        "status": "success",
        "result_state": "recommendations" if run.recommendations else "no_match",
        "user_input": user_input,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieved_candidate_count": len(results),
        "highlight_terms": _highlight_terms(user_input),
        "recommendations": recommendations,
        "warnings": [] if run.recommendations else [
            "No sufficiently supported standard was found for this requirement."
        ],
    })

    # what recommendations will have as fields(saptak)[for frontend guys]:
    '''
    {
      "standard_number": "IS XXXX",
      "title": null,
      "section_number": "5.2",
      "section_title": "Steel Pipes",
      "text": "Actual chunk of text from the standard...",
      "score": 0.91,
      "rank": 1,
      "source": {
        "chunk_id": "chunk_abc123",
        "source_file": "SP_7_2026.json",
        "page_start": 12,
        "page_end": 14,
        "element_ids": []
      }
    }
    '''

    #what source will have(saptak)[for frontend guys]:
    '''
    chunk_id
    source_file
    page_start
    page_end
    element_ids
    '''


def _serialize_recommendations(user_input, recommendations):
    """Attach optional Qwen explanations without mutating Recommendations."""
    payloads = []
    for recommendation in recommendations:
        payload = recommendation.to_dict()
        outcome = qwen_reasoner.explain(user_input, recommendation)
        payload["explanation"] = outcome.explanation
        if isinstance(outcome.explanation, dict) and outcome.explanation.get("summary"):
            payload["explanation_text"] = outcome.explanation["summary"]
        if outcome.warning:
            payload["warnings"].append(outcome.warning)
        payloads.append(payload)
    return payloads


def _highlight_terms(user_input):
    """Return meaningful query tokens for client-side title highlighting."""
    stop_words = {
        "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
    }
    return list(dict.fromkeys(
        word.strip(".,:;()[]{}")
        for word in user_input.split()
        if len(word.strip(".,:;()[]{}")) > 2
        and word.strip(".,:;()[]{}").lower() not in stop_words
    ))

# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=os.getenv("SPEC_CIPHER_HOST", "127.0.0.1"),
        port=int(os.getenv("SPEC_CIPHER_PORT", "5000")),
        debug=False,
        use_reloader=False,
    )