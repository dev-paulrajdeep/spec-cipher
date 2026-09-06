from flask import Flask, send_from_directory, request, jsonify
import os
import logging
import sqlite3
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Frontend folder is beside the backend folder
FRONTEND_DIR = os.path.join(app.root_path, "../frontend")

# Database is inside corpus/database/
DB_FILE = Path(app.root_path).parent / "corpus" / "database" / "standards.db"


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

    # --------------------------------------------------
    # Connect to SQLite database
    # --------------------------------------------------

    try:
        connection = sqlite3.connect(DB_FILE)
        cursor = connection.cursor()

        # Search the actual chunk text
        cursor.execute("""
            SELECT
                chunk_id,
                page_start,
                page_end,
                section_number,
                section_title,
                text
            FROM chunks
            WHERE text LIKE ?
            LIMIT 10
        """, (f"%{user_input}%",))

        rows = cursor.fetchall()

        connection.close()

    except sqlite3.Error as error:
        logging.error(f"Database error: {error}")

        return jsonify({
            "status": "error",
            "message": "Database error occurred."
        }), 500


    # --------------------------------------------------
    # Convert database rows to JSON
    # --------------------------------------------------

    recommendations = []

    for row in rows:

        recommendations.append({
            "chunk_id": row[0],
            "page_start": row[1],
            "page_end": row[2],
            "section_number": row[3],
            "section_title": row[4],
            "text": row[5]
        })


    # --------------------------------------------------
    # Send result back to frontend
    # --------------------------------------------------

    return jsonify({
        "status": "success",
        "user_input": user_input,
        "recommendations": recommendations,
        "warnings": []
    })


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)