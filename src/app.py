import os
import joblib
from flask import Flask, jsonify, request, render_template
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np

from .build_metadata import build_metadata
from .dataset_downloader import ensure_dataset
from .config import METADATA_CSV, MODELS_DIR, ROOT
from .feature_extractor import extract_features

# --- App Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# --- Load Trained Models (once at startup) ---
try:
    model = joblib.load(MODELS_DIR / "emotion_classifier.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    print("[INFO] Pre-trained models loaded successfully.")
except FileNotFoundError:
    print("[WARN] Model files not found. Run train_model.py to create them.")
    model = None
    scaler = None
    le = None

# --- Frontend Route ---
@app.route("/")
def home():
    """Renders the main upload page."""
    return render_template("index.html")

# --- API Endpoints ---
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/ensure_dataset")
def ensure_and_build():
    download_url = os.getenv("DOWNLOAD_URL")
    try:
        wav_root = ensure_dataset(download_url)
        df = build_metadata(download_url, out_csv=Path(METADATA_CSV))
        return jsonify({"status": "ready", "wav_root": str(wav_root), "n_files": len(df)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to classify the emotion of an uploaded audio file."""
    if model is None or scaler is None or le is None:
        return jsonify({"error": "Model is not loaded. Train the model first."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 1. Extract features
        features = extract_features(filepath)
        if features is None:
            return jsonify({"error": "Could not process audio file"}), 400

        # 2. Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))

        # 3. Predict
        prediction_encoded = model.predict(features_scaled)

        # 4. Decode prediction
        prediction_label = le.inverse_transform(prediction_encoded)[0]

        return jsonify({
            "filename": filename,
            "predicted_emotion": prediction_label
        })

    return jsonify({"error": "An unknown error occurred"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
