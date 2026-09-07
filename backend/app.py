from flask import Flask, send_from_directory, request, jsonify
import os
import sys
import logging
from pathlib import Path
# CHANGING ROOT PATH THAT PYTHON USES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.service import RetrievalService, RetrievalUnavailableError

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
    except ValueError as error:
        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400
    except Exception as error:
        logging.error(f"Retrieval error: {error}")

        return jsonify({
            "status": "error",
            "message": "Retrieval service unavailable."
        }), 500

    return jsonify({
        "status": "success",
        "user_input": user_input,
        "recommendations": [result.to_dict() for result in results],
        "warnings": []
    })


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)