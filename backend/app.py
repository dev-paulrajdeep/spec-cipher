from flask import Flask, send_from_directory, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory("../frontend", filename)


@app.route("/get_result", methods=["POST"])
def analyse_input():
    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON data received."
        }), 400

    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({
            "status": "error",
            "message": "user_input is required."
        }), 400

    print("user r input", user_input)   # used it for debbuging ,, siddhartha 
   

    return jsonify({
        "status": "success",
        "user_input": user_input,
        "recommendations": [],
        "warnings": []
    })


if __name__ == "__main__":
    app.run(debug=True)
