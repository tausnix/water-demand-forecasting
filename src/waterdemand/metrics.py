"""Forecast metrics.

sMAPE follows the reference implementation's convention (200 * |e| / (|y|+|yhat|)),
which keeps our numbers directly comparable to the published artefacts.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = float(
        np.mean(np.where(denominator == 0, 0.0, 200.0 * np.abs(y_true - y_pred) / denominator))
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": smape,
    }


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE reduction vs a baseline. 0.44 means 44% better than naive."""
    return 1.0 - model_mae / baseline_mae
