# ─────────────────────────────────────────────────────────────
#  train.py  –  Run once to train every model for every disease
#  Usage:  python train.py
#          python train.py --disease cardiovascular
# ─────────────────────────────────────────────────────────────

import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# local imports (run from backend/)
from config              import DISEASE_REGISTRY
from services.data_generator import generate
from services.preprocessor   import Preprocessor
from services.evaluator      import evaluate
from models.model_definitions import get_all_models
from models.model_loader      import save, artifact_name


def train_disease(disease: str):
    print(f"\n{'='*58}")
    print(f"  Training models for:  {disease.upper()}")
    print(f"{'='*58}")

    # ── 1. Data ───────────────────────────────────────────────
    df      = generate(disease, n=3000)
    X       = df.drop("target", axis=1).values
    y       = df["target"].values
    feat_names = df.drop("target", axis=1).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 2. Preprocessing ──────────────────────────────────────
    pre = Preprocessor()
    X_train_s = pre.fit_transform(X_train)
    X_test_s  = pre.transform(X_test)

    # Save scaler
    save(pre, artifact_name(disease, "scaler"))
    # Save feature names
    save(feat_names, artifact_name(disease, "features"))

    # ── 3. Train & evaluate each model ────────────────────────
    all_models  = get_all_models()
    all_metrics = {}

    for key, model in all_models.items():
        print(f"\n  ▸ {key}")
        model.fit(X_train_s, y_train)
        metrics = evaluate(model, X_test_s, y_test)
        all_metrics[key] = metrics
        save(model, artifact_name(disease, key))
        print(f"    acc={metrics['accuracy']:.4f}  "
              f"prec={metrics['precision']:.4f}  "
              f"rec={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}")

    # Save comparison metrics
    save(all_metrics, artifact_name(disease, "metrics"))
    print(f"\n  ✅  All models saved to model_store/")
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default=None,
                        help="Train a single disease (default: all)")
    args = parser.parse_args()

    diseases = ([args.disease] if args.disease
                else list(DISEASE_REGISTRY.keys()))

    for d in diseases:
        if d not in DISEASE_REGISTRY:
            print(f"Unknown disease '{d}'. Check config.py.")
            sys.exit(1)
        train_disease(d)

    print("\n🎉  Training complete.")


if __name__ == "__main__":
    main()
