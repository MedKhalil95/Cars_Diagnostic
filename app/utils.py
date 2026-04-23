"""
utils.py — shared preprocessing helpers used by train.py and app.py
"""

from __future__ import annotations
import numpy as np
import joblib
import torch
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent


def load_artifacts(model_dir: Path = MODEL_DIR):
    """
    Load all persisted artefacts produced by train.py.
    Returns (model, brand_enc, model_enc, engine_enc, health_enc, feature_cols).
    """
    from model.network import CarDiagnosticNet

    brand_enc    = joblib.load(model_dir / "brand_encoder.pkl")
    model_enc    = joblib.load(model_dir / "model_encoder.pkl")
    engine_enc   = joblib.load(model_dir / "engine_encoder.pkl")
    health_enc   = joblib.load(model_dir / "health_encoder.pkl")
    feature_cols = joblib.load(model_dir / "feature_cols.pkl")

    num_classes = len(health_enc.classes_)
    net = CarDiagnosticNet(input_size=len(feature_cols), num_classes=num_classes)
    state = torch.load(str(model_dir / "model.pt"), map_location=torch.device("cpu"))
    net.load_state_dict(state)
    net.eval()

    return net, brand_enc, model_enc, engine_enc, health_enc, feature_cols


def build_feature_vector(
    form_data: dict,
    brand_enc,
    model_enc,
    engine_enc,
    feature_cols: list[str],
) -> torch.Tensor:
    """
    Convert raw form values into a float32 tensor in the exact column order
    used during training. Raises ValueError on unknown categories or missing fields.
    """
    row: dict[str, float] = {}

    for col in feature_cols:
        raw = form_data.get(col)
        if raw is None or raw == "":
            raise ValueError(f"Missing field: {col}")

        if col == "brand":
            row[col] = float(brand_enc.transform([raw])[0])
        elif col == "model":
            row[col] = float(model_enc.transform([raw])[0])
        elif col == "engine_type":
            row[col] = float(engine_enc.transform([raw])[0])
        else:
            row[col] = float(raw)

    vec = np.array([[row[c] for c in feature_cols]], dtype=np.float32)
    return torch.tensor(vec)


def decode_prediction(pred_idx: int, health_enc) -> str:
    """Map integer class index back to a human-readable label."""
    return health_enc.inverse_transform([pred_idx])[0]