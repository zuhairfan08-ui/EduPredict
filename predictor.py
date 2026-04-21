"""
Prediction utilities for the AI Student Performance Predictor.

Loads the trained models (training them on first use if missing)
and exposes a clean predict() API used by the Streamlit app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

from train_model import FEATURES, MODEL_DIR, train_and_save


@dataclass
class PredictionResult:
    predicted_marks: float
    pass_probability: float
    will_pass: bool
    performance_label: str
    explanation: str


def _category(marks: float) -> str:
    if marks < 40:
        return "Fail"
    if marks < 60:
        return "Average"
    if marks < 80:
        return "Good"
    return "Excellent"


def _ensure_models() -> None:
    needed = ["regressor.joblib", "classifier.joblib", "scaler.joblib", "metrics.joblib"]
    if not all(os.path.exists(os.path.join(MODEL_DIR, n)) for n in needed):
        train_and_save()


def load_artifacts() -> tuple:
    _ensure_models()
    reg = joblib.load(os.path.join(MODEL_DIR, "regressor.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    metrics = joblib.load(os.path.join(MODEL_DIR, "metrics.joblib"))
    return reg, clf, scaler, metrics


def predict(
    study_hours: float,
    attendance: float,
    sleep_hours: float,
    assignments_completed: int,
    stress_level: int,
) -> PredictionResult:
    reg, clf, scaler, _ = load_artifacts()

    X = np.array(
        [[study_hours, attendance, sleep_hours, assignments_completed, stress_level]],
        dtype=float,
    )
    Xs = scaler.transform(X)

    marks = float(np.clip(reg.predict(Xs)[0], 0, 100))
    proba = float(clf.predict_proba(Xs)[0][1])
    will_pass = bool(proba >= 0.5)
    label = _category(marks)

    # Simple human-readable explanation based on the strongest signals.
    notes = []
    if study_hours < 3:
        notes.append("study time is quite low")
    elif study_hours >= 7:
        notes.append("strong daily study time")
    if attendance < 60:
        notes.append("attendance is low")
    elif attendance >= 85:
        notes.append("excellent attendance")
    if sleep_hours < 5 or sleep_hours > 9:
        notes.append("sleep schedule is off-balance")
    if stress_level >= 8:
        notes.append("stress level is very high")
    if assignments_completed <= 3:
        notes.append("few assignments completed")

    if not notes:
        explanation = "Inputs look balanced overall."
    else:
        explanation = "Key factors: " + ", ".join(notes) + "."

    return PredictionResult(
        predicted_marks=round(marks, 2),
        pass_probability=round(proba, 4),
        will_pass=will_pass,
        performance_label=label,
        explanation=explanation,
    )
