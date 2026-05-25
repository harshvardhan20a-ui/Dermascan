"""
app.py - DermaScan AI Flask Backend
=====================================
Routes:
    GET  /          -> Serves the main web UI (index.html)
    POST /analyze   -> Receives uploaded image, runs ML model, returns JSON

Setup:
    1. pip install -r requirements.txt
    2. python train1.py             (trains and saves the model)
    3. python app.py               (starts the web server)
    4. Open http://localhost:5000

The model is loaded lazily on first request from predict.py.
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()   # Load variables from .env file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    -------------
    Expects multipart/form-data with:
        image    : image file (JPG / PNG / WebP)
        symptoms : comma-separated string of selected symptoms
        location : user's city/region string

    Returns JSON with ML prediction and recommendations.
    """
    # Validate file
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload JPG, PNG, or WebP."}), 400

    symptoms    = request.form.get("symptoms", "None reported")
    location    = request.form.get("location", "India")
    image_bytes = file.read()

    try:
        from predict import predict
        result = predict(image_bytes, location=location, symptoms=symptoms)
        return jsonify(result)

    except FileNotFoundError as e:
        # Model hasn't been trained yet
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(_):
    return jsonify({"error": "File too large. Maximum allowed size is 10 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    print(f"\nDermaScan AI starting on http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
