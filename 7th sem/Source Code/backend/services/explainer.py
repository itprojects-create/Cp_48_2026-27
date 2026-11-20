# ─────────────────────────────────────────────────────────────
#  services/explainer.py
#  Returns top contributing features for a prediction.
#  Currently uses model-native feature importances.
#  Replace _shap_explain() with real SHAP when needed:
#    pip install shap
#    import shap; explainer = shap.TreeExplainer(model); ...
# ─────────────────────────────────────────────────────────────

import numpy as np


def top_features(model, feature_names: list, input_vector: np.ndarray,
                 top_n: int = 3) -> list:
    """
    Returns top_n features most responsible for this prediction.
    Strategy:
      - Tree-based models  → native feature_importances_
      - Linear models      → abs(coef_)
      - Ensemble           → average of member importances
      - Fallback           → equal weight (simulated)
    """
    importances = _get_importances(model, feature_names)
    ranked = sorted(zip(feature_names, importances),
                    key=lambda x: x[1], reverse=True)
    return [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in ranked[:top_n]
    ]


def _get_importances(model, feature_names: list) -> list:
    # Tree-based (RF, DT, XGB)
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist()
    # Linear (LR)
    if hasattr(model, "coef_"):
        return np.abs(model.coef_[0]).tolist()
    # Ensemble VotingClassifier — average sub-estimator importances
    if hasattr(model, "estimators_"):
        imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                imps.append(np.abs(est.coef_[0]))
        if imps:
            return np.mean(imps, axis=0).tolist()
    # Fallback: equal weight
    n = len(feature_names)
    return [1.0 / n] * n


# ── SHAP placeholder ──────────────────────────────────────────
def _shap_explain(model, input_vector: np.ndarray):
    """
    Uncomment when shap is installed:

    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_vector)
    return shap_values
    """
    raise NotImplementedError("Install shap and uncomment _shap_explain body")
