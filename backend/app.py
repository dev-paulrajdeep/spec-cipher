from flask import Flask, send_from_directory, request, jsonify
import os
import sys
import logging
from pathlib import Path
# CHANGING ROOT PATH THAT PYTHON USES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.service import RetrievalService, RetrievalUnavailableError
from retrieval.indexing import IndexCompatibilityError

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Frontend folder is beside the backend folder
FRONTEND_DIR = os.path.join(app.root_path, "../frontend")

# 


# Generated retrieval bundle; build it before starting Flask.
RETRIEVAL_INDEX_DIR = Path(app.root_path).parent / "retrieval-index"
retriever = RetrievalService.from_persisted(RETRIEVAL_INDEX_DIR)


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
        results = retriever.search(user_input, top_k=5)

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
        "recommendations": [result.to_dict() for result in results],
        "warnings": []
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


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)