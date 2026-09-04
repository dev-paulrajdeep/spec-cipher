from flask import Flask, send_from_directory, request

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
    user_input = data["user_input"]

    return f"testing...get_result...connection: {user_input}"

if __name__ == "__main__":
    app.run(debug=True)