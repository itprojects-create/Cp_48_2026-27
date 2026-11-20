# ─────────────────────────────────────────────────────────────
#  services/evaluator.py
#  Computes accuracy, precision, recall, F1 for a model.
# ─────────────────────────────────────────────────────────────

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
    }
