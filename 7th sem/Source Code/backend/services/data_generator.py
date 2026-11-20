import numpy as np
import pandas as pd

def generate(disease: str, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    generators = {
        "cardiovascular": _cardiovascular,
        "diabetes":       _diabetes,
    }
    if disease not in generators:
        raise ValueError(f"No data generator for disease '{disease}'")
    return generators[disease](n)

def _cardiovascular(n):
    age            = np.random.randint(20, 80, n)
    blood_pressure = np.random.randint(60, 180, n)
    cholesterol    = np.random.randint(100, 350, n)
    bmi            = np.round(np.random.uniform(15.0, 45.0, n), 1)
    glucose        = np.random.randint(70, 300, n)
    smoking        = np.random.randint(0, 2, n)
    family_history = np.random.randint(0, 2, n)
    risk_score = (
        (age > 50).astype(int) * 2 +
        (blood_pressure > 130).astype(int) * 2 +
        (cholesterol > 240).astype(int) * 2 +
        (bmi > 30).astype(int) * 1 +
        (glucose > 140).astype(int) * 2 +
        smoking * 2 + family_history * 1 +
        np.random.randint(0, 3, n)
    )
    return pd.DataFrame({
        "age": age, "blood_pressure": blood_pressure,
        "cholesterol": cholesterol, "bmi": bmi,
        "glucose": glucose, "smoking": smoking,
        "family_history": family_history,
        "target": (risk_score >= 6).astype(int),
    })

def _diabetes(n):
    pregnancies    = np.random.randint(0, 15, n)
    glucose        = np.random.randint(60, 200, n)
    blood_pressure = np.random.randint(40, 130, n)
    skin_thickness = np.random.randint(5, 60, n)
    insulin        = np.random.randint(0, 500, n)
    bmi            = np.round(np.random.uniform(18.0, 55.0, n), 1)
    dpf            = np.round(np.random.uniform(0.05, 2.5, n), 3)
    age            = np.random.randint(18, 75, n)
    risk_score = (
        (glucose > 125).astype(int) * 3 +
        (bmi > 30).astype(int) * 2 +
        (age > 45).astype(int) * 2 +
        (insulin > 200).astype(int) * 1 +
        (dpf > 0.5).astype(int) * 1 +
        (pregnancies > 3).astype(int) * 1 +
        (blood_pressure > 90).astype(int) * 1 +
        np.random.randint(0, 3, n)
    )
    return pd.DataFrame({
        "age": age, "glucose": glucose,
        "bmi": bmi, "blood_pressure": blood_pressure,
        "insulin": insulin, "skin_thickness": skin_thickness,
        "pregnancies": pregnancies,
        "diabetes_pedigree_function": dpf,
        "target": (risk_score >= 5).astype(int),
    })
