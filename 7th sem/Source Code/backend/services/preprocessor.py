# ─────────────────────────────────────────────────────────────
#  services/preprocessor.py
#  Handles feature scaling. Scaler is fitted once during
#  training and reused at prediction time.
# ─────────────────────────────────────────────────────────────

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)
