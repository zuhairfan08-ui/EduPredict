"""
Train Linear Regression (marks prediction) and Logistic Regression
(Pass/Fail classification) for the AI Student Performance Predictor.

Run:
    python train_model.py

Outputs:
    models/regressor.joblib
    models/classifier.joblib
    models/scaler.joblib
    models/metrics.joblib
"""

from __future__ import annotations

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generator import generate_dataset, save_dataset

FEATURES = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "assignments_completed",
    "stress_level",
]

# ── Use /tmp on Streamlit Cloud (read-only src mount), local "models" otherwise
MODEL_DIR = os.path.join(tempfile.gettempdir(), "edupredict_models")
DATA_PATH = "data/students.csv"


def load_or_create_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    save_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def train_and_save() -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_or_create_data()

    # Basic cleaning: drop any nulls (synthetic data shouldn't have, but safe).
    df = df.dropna(subset=FEATURES + ["marks", "passed"])

    X = df[FEATURES].values
    y_reg = df["marks"].values
    y_clf = df["passed"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X_scaled, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Regression
    reg = LinearRegression()
    reg.fit(X_train, yr_train)
    yr_pred = reg.predict(X_test)
    r2 = float(r2_score(yr_test, yr_pred))
    mae = float(mean_absolute_error(yr_test, yr_pred))

    # Classification
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, yc_train)
    yc_pred = clf.predict(X_test)
    acc = float(accuracy_score(yc_test, yc_pred))

    metrics = {
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "classification_accuracy": round(acc, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
    }

    joblib.dump(reg,     os.path.join(MODEL_DIR, "regressor.joblib"))
    joblib.dump(clf,     os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics.joblib"))

    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print("Training complete.")
    print(f"  Models saved to: {MODEL_DIR}")
    for k, v in m.items():
        print(f"  {k}: {v}")
