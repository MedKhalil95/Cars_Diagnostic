"""
app.py — Flask web application for Car Engine Health Diagnosis
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_artifacts, build_feature_vector, decode_prediction
from generate_data import BRAND_MODEL_NAMES   # brand → [model, ...] mapping

app = Flask(__name__)

# ── Load model & encoders once at startup ────────────────────────────────────
MODEL_DIR = ROOT / "model"
try:
    model, brand_enc, model_enc, engine_enc, health_enc, feature_cols = load_artifacts(MODEL_DIR)
    print(f"✅ Model loaded  |  features={feature_cols}  |  classes={list(health_enc.classes_)}")
except FileNotFoundError as e:
    raise SystemExit(f"❌ Could not load model artefacts: {e}\n"
                     "   Run train.py first to generate model.pt and *.pkl files.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        brands=list(brand_enc.classes_),
        engine_types=list(engine_enc.classes_),
        # JSON map consumed by JS for dynamic model dropdown
        brand_models_json=json.dumps(BRAND_MODEL_NAMES),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict() if request.form else request.get_json(force=True)

        import torch
        features = build_feature_vector(form_data, brand_enc, model_enc, engine_enc, feature_cols)

        with torch.no_grad():
            logits = model(features)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(torch.argmax(probs).item())

        label      = decode_prediction(idx, health_enc)
        confidence = round(float(probs[idx]) * 100, 1)

        return jsonify({"prediction": label, "confidence": confidence})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)