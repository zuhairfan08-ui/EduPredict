"""
Synthetic dataset generator for the AI Student Performance Predictor.

Creates a realistic dataset of student study habits and resulting marks.
The relationship between features and marks is intentionally non-trivial
so that ML models have something meaningful to learn.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


RANDOM_STATE = 42


def generate_dataset(n_samples: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic student performance dataset.

    Features:
        - study_hours          (0 - 12)
        - attendance           (40 - 100 %)
        - sleep_hours          (3 - 10)
        - assignments_completed(0 - 10)
        - stress_level         (1 - 10)

    Targets:
        - marks       (0 - 100)
        - performance (Fail / Average / Good / Excellent)
        - passed      (0 / 1)
    """
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(5, 2.2, n_samples), 0, 12)
    attendance = np.clip(rng.normal(78, 12, n_samples), 40, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.3, n_samples), 3, 10)
    assignments = np.clip(rng.normal(7, 2.0, n_samples), 0, 10).round()
    stress = np.clip(rng.normal(5, 2.0, n_samples), 1, 10).round()

    # Sleep penalty: too little or too much sleep both hurt performance.
    sleep_penalty = -1.5 * (sleep_hours - 7.5) ** 2

    marks = (
        5.2 * study_hours
        + 0.35 * attendance
        + sleep_penalty
        + 1.4 * assignments
        - 1.1 * stress
        + rng.normal(0, 4, n_samples)
        + 5  # base
    )
    marks = np.clip(marks, 0, 100)

    def to_label(m: float) -> str:
        if m < 40:
            return "Fail"
        if m < 60:
            return "Average"
        if m < 80:
            return "Good"
        return "Excellent"

    performance = np.array([to_label(m) for m in marks])
    passed = (marks >= 40).astype(int)

    return pd.DataFrame(
        {
            "study_hours": study_hours.round(2),
            "attendance": attendance.round(2),
            "sleep_hours": sleep_hours.round(2),
            "assignments_completed": assignments.astype(int),
            "stress_level": stress.astype(int),
            "marks": marks.round(2),
            "performance": performance,
            "passed": passed,
        }
    )


def save_dataset(path: str = "data/students.csv", n_samples: int = 1000) -> str:
    df = generate_dataset(n_samples=n_samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = save_dataset()
    print(f"Dataset saved to: {out}")
