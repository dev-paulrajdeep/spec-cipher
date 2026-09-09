from flask import Flask, send_from_directory, request, jsonify
import atexit
import os
import sys
import logging
from pathlib import Path
# CHANGING ROOT PATH THAT PYTHON USES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.service import RetrievalService, RetrievalUnavailableError
from retrieval.indexing import IndexCompatibilityError 
from retrieval.config import RetrievalConfig
from intelligence.pipeline import IntelligenceEngine
from intelligence.relationships import CorpusKnowledgeBase
from backend.qwen_reasoning import QwenReasoner

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Frontend folder is beside the backend folder
FRONTEND_DIR = os.path.join(app.root_path, "../frontend")

# Generated retrieval bundle; build it before starting Flask.
RETRIEVAL_INDEX_DIR = Path(app.root_path).parent / "retrieval-index"
retriever = RetrievalService.from_persisted(
    RETRIEVAL_INDEX_DIR,
    config=RetrievalConfig(reranker_enabled=True, sequential_model_lifecycle=False),
)
close_retriever = getattr(retriever, "close", None)
if callable(close_retriever):
    atexit.register(close_retriever)

CANONICAL_CORPUS_PATHS = (
    PROJECT_ROOT / "corpus" / "data" / "SP 7_2026.json",
    PROJECT_ROOT / ".tmp" / "sp7_validation" / "SP 7_2026.json",
)
canonical_corpus_file = next((p for p in CANONICAL_CORPUS_PATHS if p.exists()), None)
knowledge_base = (
    CorpusKnowledgeBase.from_canonical_json((canonical_corpus_file,))
    if canonical_corpus_file
    else CorpusKnowledgeBase()
)
intelligence = IntelligenceEngine(knowledge_base=knowledge_base)
qwen_reasoner = QwenReasoner()



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
            results = retriever.search(user_input, top_k=5)
            run = intelligence.recommend(user_input, results)
            recommendations = _serialize_recommendations(user_input, run.recommendations)

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
        "message": "Retrieval service unavailable."
        }), 503

    except IndexCompatibilityError as error:
        logging.exception("Retrieval index is incompatible")
        return jsonify({
        "status": "error",
        "message": "Retrieval service unavailable."
        }), 503

    except Exception as error:
        logging.exception("Recommendation pipeline failed")
        return jsonify({
            "status": "error",
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
        "user_input": user_input,
        "recommendations": recommendations,
        "warnings": [] if run.recommendations else [
            "No suitable standard was found for this requirement."
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
        if outcome.warning:
            payload["warnings"].append(outcome.warning)
        payloads.append(payload)
    return payloads

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