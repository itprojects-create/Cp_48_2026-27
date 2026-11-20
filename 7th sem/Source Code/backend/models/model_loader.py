# ─────────────────────────────────────────────────────────────
#  models/model_loader.py
#  Persists and loads trained models + preprocessors.
# ─────────────────────────────────────────────────────────────

import os
import pickle
from config import MODEL_STORE_DIR


def save(obj, filename: str):
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    path = os.path.join(MODEL_STORE_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(filename: str):
    path = os.path.join(MODEL_STORE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run train.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def artifact_name(disease: str, key: str) -> str:
    """e.g.  'cardiovascular_random_forest.pkl'"""
    return f"{disease}_{key}.pkl"
