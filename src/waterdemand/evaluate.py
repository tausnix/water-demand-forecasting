"""Evaluation: time-aware cross-validation and per-segment error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .metrics import regression_metrics
from .models import build_pipeline
from .preprocessing import build_scale_preprocessor, build_tree_preprocessor


def time_series_cv(
    frame: pd.DataFrame,
    names: list[str],
    n_splits: int = 5,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window CV, fitted inside each fold.

    Every transform (imputation, Yeo-Johnson, scaling, encoding) is re-fitted on
    the fold's training rows only — that is the whole point of doing this inside
    the loop rather than once up front.

    Note: the reference implementation deliberately excluded the VotingRegressor
    from CV. We include it, because the model we select should have CV evidence.
    """
    folds = []
    for name in names:
        for fold, (train_idx, valid_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(frame), start=1
        ):
            tr = frame.iloc[train_idx]
            va = frame.iloc[valid_idx]
            pipeline = build_pipeline(name, params)
            pipeline.fit(tr[config.FEATURES], tr[config.TARGET])
            pred = pipeline.predict(va[config.FEATURES])
            folds.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_rows": len(tr),
                    "valid_rows": len(va),
                    **regression_metrics(va[config.TARGET], pred),
                }
            )
        print(f"[cv] {name}: done ({n_splits} folds)")

    detailed = pd.DataFrame(folds)
    summary = (
        detailed.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            smape_mean=("smape_pct", "mean"),
        )
        .sort_values("mae_mean")
        .reset_index(drop=True)
    )
    return detailed, summary


def segment_errors(
    frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """MAE broken down by hour-of-day and day-of-week.

    An aggregate MAE of 158 hides the fact that some hours are far harder than
    others. This table is what an operations team actually needs.
    """
    out = frame[[config.TIMESTAMP]].copy()
    out["hour"] = out[config.TIMESTAMP].dt.hour
    out["day_of_week"] = out[config.TIMESTAMP].dt.day_name()
    out["abs_error"] = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))

    by_hour = out.groupby("hour", as_index=False)["abs_error"].mean()
    by_hour.columns = ["hour", "mae"]
    by_hour["share_of_total_error"] = by_hour["mae"] / by_hour["mae"].sum()
    return by_hour


def residual_summary(y_true, y_pred) -> dict:
    residual = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return {
        "bias": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "p05": float(np.percentile(residual, 5)),
        "p50": float(np.percentile(residual, 50)),
        "p95": float(np.percentile(residual, 95)),
        "under_forecast_rate": float(np.mean(residual > 0)),
    }
