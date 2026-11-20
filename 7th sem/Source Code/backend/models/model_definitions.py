# ─────────────────────────────────────────────────────────────
#  models/model_definitions.py
#  Single source of truth for every ML model used.
#  Adding a new model = add one entry to get_all_models().
# ─────────────────────────────────────────────────────────────

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, VotingClassifier

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("⚠  XGBoost not installed – skipping XGBoost model")


def get_all_models() -> dict:
    """
    Returns {model_key: unfitted_model_instance}.
    The ensemble is built from the base models automatically.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "decision_tree":       DecisionTreeClassifier(max_depth=6,
                                                       random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=200,
                                                       max_depth=8,
                                                       random_state=42,
                                                       n_jobs=-1),
    }

    if _XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric="logloss", random_state=42,
        )

    # Ensemble: soft-voting across all base models
    estimators = [(k, v) for k, v in models.items()]
    models["ensemble"] = VotingClassifier(estimators=estimators,
                                          voting="soft", n_jobs=-1)
    return models
