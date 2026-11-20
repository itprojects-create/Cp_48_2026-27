import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")

SUPPORTED_MODELS = ["logistic_regression", "decision_tree",
                    "random_forest", "xgboost", "ensemble"]

DISEASE_REGISTRY = {
    "cardiovascular": {
        "display_name": "Cardiovascular Disease",
        "features": [
            {"key": "age",            "label": "Age",            "unit": "years", "min": 1,   "max": 120},
            {"key": "blood_pressure", "label": "Blood Pressure", "unit": "mmHg",  "min": 40,  "max": 250},
            {"key": "cholesterol",    "label": "Cholesterol",    "unit": "mg/dL", "min": 50,  "max": 500},
            {"key": "bmi",            "label": "BMI",            "unit": "kg/m²", "min": 10,  "max": 70},
            {"key": "glucose",        "label": "Blood Glucose",  "unit": "mg/dL", "min": 50,  "max": 500},
            {"key": "smoking",        "label": "Smoker",         "unit": "0/1",   "min": 0,   "max": 1},
            {"key": "family_history", "label": "Family History", "unit": "0/1",   "min": 0,   "max": 1},
        ],
    },
    "diabetes": {
        "display_name": "Diabetes",
        "features": [
            {"key": "age",                        "label": "Age",            "unit": "years",  "min": 1,   "max": 120},
            {"key": "glucose",                    "label": "Glucose",        "unit": "mg/dL",  "min": 50,  "max": 500},
            {"key": "bmi",                        "label": "BMI",            "unit": "kg/m²",  "min": 10,  "max": 70},
            {"key": "blood_pressure",             "label": "Blood Pressure", "unit": "mmHg",   "min": 40,  "max": 250},
            {"key": "insulin",                    "label": "Insulin",        "unit": "µU/mL",  "min": 0,   "max": 900},
            {"key": "skin_thickness",             "label": "Skin Thickness", "unit": "mm",     "min": 0,   "max": 100},
            {"key": "pregnancies",                "label": "Pregnancies",    "unit": "count",  "min": 0,   "max": 20},
            {"key": "diabetes_pedigree_function", "label": "DPF",            "unit": "score",  "min": 0,   "max": 2.5},
        ],
    },
}

RISK_LEVELS = [
    {"label": "Low",       "max": 0.30, "color": "#22c55e"},
    {"label": "Moderate",  "max": 0.60, "color": "#f59e0b"},
    {"label": "High",      "max": 0.80, "color": "#f97316"},
    {"label": "Very High", "max": 1.01, "color": "#ef4444"},
]

def get_risk(prob: float) -> dict:
    for r in RISK_LEVELS:
        if prob <= r["max"]:
            return r
    return RISK_LEVELS[-1]
