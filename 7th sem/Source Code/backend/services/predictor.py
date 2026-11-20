# ─────────────────────────────────────────────────────────────
#  services/predictor.py
#  Orchestrates: load → preprocess → predict → explain
# ─────────────────────────────────────────────────────────────

import numpy as np
from config              import get_risk, DISEASE_REGISTRY
from models.model_loader import load, artifact_name
from services.explainer  import top_features


def predict(disease: str, raw_inputs: dict, model_key: str = "ensemble") -> dict:
    """
    Main prediction entry point.
    Returns a unified response dict consumed by the API route.
    """
    # ── Load artefacts ────────────────────────────────────────
    model      = load(artifact_name(disease, model_key))
    scaler     = load(artifact_name(disease, "scaler"))
    feat_names = load(artifact_name(disease, "features"))

    # ── Build & validate input vector ────────────────────────
    vector = _build_vector(raw_inputs, feat_names, disease)
    vector_scaled = scaler.transform(vector)

    # ── Predict ───────────────────────────────────────────────
    prob       = float(model.predict_proba(vector_scaled)[0][1])
    prediction = int(model.predict(vector_scaled)[0])
    risk       = get_risk(prob)

    # ── Explain ───────────────────────────────────────────────
    contributors = top_features(model, feat_names, vector_scaled, top_n=3)

    return {
        "disease":      disease,
        "model_used":   model_key,
        "prediction":   prediction,
        "probability":  round(prob, 4),
        "percent":      round(prob * 100, 1),
        "risk_level":   risk["label"],
        "risk_color":   risk["color"],
        "top_features": contributors,
        "advice":       _advice(risk["label"]),
        "inputs":       raw_inputs,
    }


def compare_models(disease: str, raw_inputs: dict) -> dict:
    """Returns predictions from every available model for comparison."""
    metrics = load(artifact_name(disease, "metrics"))
    results = {}
    for key in metrics.keys():
        try:
            r = predict(disease, raw_inputs, model_key=key)
            results[key] = {
                "probability": r["probability"],
                "percent":     r["percent"],
                "risk_level":  r["risk_level"],
                "metrics":     metrics[key],
            }
        except Exception as e:
            results[key] = {"error": str(e)}
    return {"disease": disease, "models": results}


# ── Helpers ───────────────────────────────────────────────────

def _build_vector(raw: dict, feat_names: list, disease: str) -> np.ndarray:
    config_feats = {f["key"]: f for f in DISEASE_REGISTRY[disease]["features"]}
    row = []
    for feat in feat_names:
        if feat not in raw:
            raise ValueError(f"Missing input feature: '{feat}'")
        val = float(raw[feat])
        cfg = config_feats.get(feat, {})
        if "min" in cfg and "max" in cfg:
            if not (cfg["min"] <= val <= cfg["max"]):
                raise ValueError(
                    f"'{feat}' must be {cfg['min']}–{cfg['max']}, got {val}"
                )
        row.append(val)
    return np.array([row])


_ADVICE = {
    "Low":       "Maintain your healthy lifestyle. Regular annual check-ups are still recommended.",
    "Moderate":  "Consider dietary improvements and increased physical activity. Consult a doctor.",
    "High":      "Please consult a healthcare professional soon for a thorough evaluation.",
    "Very High": "Seek medical attention promptly. Immediate lifestyle changes are essential.",
}

def _advice(level: str) -> str:
    return _ADVICE.get(level, "Consult a healthcare professional.")
