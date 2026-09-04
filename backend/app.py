from flask import Flask, send_from_directory, request, jsonify
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

FRONTEND_DIR = os.path.join(app.root_path, "../frontend")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def frontend_files(filename):
    # allowed = {"css", "js", "png", "jpg", "html"}
     #if filename.split(".")[-1] not in allowed:
      #  return jsonify({"status": "error", "message": "File type not allowed"}), 403
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/get_result", methods=["POST"])
def analyse_input():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received."}), 400

    user_input = data.get("user_input", "").strip()
    if not user_input:
        return jsonify({"status": "error", "message": "user_input is required."}), 400

    logging.debug(f"user_input: {user_input}")  # used it for debugging ,, siddhartha

    return jsonify({
        "status": "success",
        "user_input": user_input,
        "recommendations": [],
        "warnings": []
    })

if __name__ == "__main__":
    app.run(debug=True)
