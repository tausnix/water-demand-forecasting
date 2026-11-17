"""Training stages: baselines -> ensembles -> selection -> locked test.

The ordering here is the methodological claim of the project:

    1. `baseline_stage`   — persistence + Ridge + RandomForest on the validation block
    2. `ensemble_stage`   — Bagging / HGB / XGBoost / Voting on the validation block
    3. (tuning)           — Optuna and Keras Tuner, scored on validation only
    4. `select`           — pick on validation, write reports/selection.json
    5. `locked_test`      — read the test block exactly once, refit on train+val

Nothing before step 5 is allowed to look at `test`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config, models
from .metrics import regression_metrics

BASELINE_MODELS = (
    "ridge_yj_robust",
    "random_forest",
)


def _fit_score(name: str, train, eval_frame, params: dict | None = None):
    pipeline = models.build_pipeline(name, params)
    started = time.perf_counter()
    pipeline.fit(train[config.FEATURES], train[config.TARGET])
    elapsed = time.perf_counter() - started
    pred = pipeline.predict(eval_frame[config.FEATURES])
    return pred, elapsed, pipeline


def baseline_stage(train, validation) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persistence baselines plus the two cheapest learned models."""
    y_true = validation[config.TARGET]
    rows = []
    preds = pd.DataFrame(
        {config.TIMESTAMP: validation[config.TIMESTAMP], "actual": y_true.to_numpy()}
    )

    for name, metrics in models.naive_scores(validation).items():
        lag = int(name.rsplit("_", 1)[1])
        preds[name] = models.persistence(validation, lag)
        rows.append({"model": name, "stage": "baseline", "fit_seconds": 0.0, **metrics})

    for name in BASELINE_MODELS:
        pred, elapsed, _ = _fit_score(name, train, validation)
        preds[name] = pred
        rows.append(
            {
                "model": name,
                "stage": "baseline",
                "fit_seconds": elapsed,
                **regression_metrics(y_true, pred),
            }
        )

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return table, preds


def ensemble_stage(train, validation, params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The four ensemble-family models, fitted on train, scored on validation."""
    y_true = validation[config.TARGET]
    rows = []
    preds = pd.DataFrame(
        {config.TIMESTAMP: validation[config.TIMESTAMP], "actual": y_true.to_numpy()}
    )

    for name in (
        "bagging_extra_tree",
        "hist_gradient_boosting",
        "xgboost",
        "voting_regressor",
    ):
        pred, elapsed, _ = _fit_score(name, train, validation, params)
        preds[name] = pred
        rows.append(
            {
                "model": name,
                "stage": "ensemble",
                "fit_seconds": elapsed,
                **regression_metrics(y_true, pred),
            }
        )
        print(f"[ensemble] {name}: MAE={rows[-1]['mae']:.3f}")

    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    return table, preds


def select(validation_table: pd.DataFrame, tuned_rows: dict | None = None) -> dict:
    """Freeze the choice on validation evidence and persist it."""
    table = validation_table.copy()
    if tuned_rows:
        table = pd.concat([table, pd.DataFrame(tuned_rows)], ignore_index=True)

    table = table.sort_values("mae").reset_index(drop=True)
    winner = table.iloc[0].to_dict()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_before_test": winner["model"],
        "locked_test_used_for_selection": False,
        "selection_metric": "validation MAE",
        "validation_mae": winner["mae"],
        "runner_up_margin": (
            float(table.iloc[1]["mae"] - winner["mae"]) if len(table) > 1 else None
        ),
        "candidates": int(len(table)),
    }
    config.SELECTION_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def locked_test(
    train,
    validation,
    test,
    name: str,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The one and only evaluation on unseen data.

    Refits on train+validation, which is legitimate: those rows are all in the
    past relative to the test block.
    """
    if not config.SELECTION_JSON.exists():
        raise RuntimeError(
            "refusing to open the test block before reports/selection.json exists"
        )
    frozen = json.loads(config.SELECTION_JSON.read_text(encoding="utf-8"))
    if frozen["selected_before_test"] != name:
        raise RuntimeError(
            f"selection was frozen to {frozen['selected_before_test']!r}, not {name!r}"
        )

    development = pd.concat([train, validation], ignore_index=True)
    y_true = test[config.TARGET]

    rows = []
    preds = pd.DataFrame(
        {config.TIMESTAMP: test[config.TIMESTAMP], "actual": y_true.to_numpy()}
    )

    # persistence context is always reported alongside the model
    for naive, metrics in models.naive_scores(test).items():
        lag = int(naive.rsplit("_", 1)[1])
        preds[naive] = models.persistence(test, lag)
        rows.append({"model": naive, "role": "baseline", **metrics})

    pipeline = models.build_pipeline(name, params)
    pipeline.fit(development[config.FEATURES], development[config.TARGET])
    pred = pipeline.predict(test[config.FEATURES])
    preds[name] = pred
    metrics = regression_metrics(y_true, pred)
    rows.append({"model": name, "role": "selected", **metrics})

    table = pd.DataFrame(rows).sort_values(["role", "mae"]).reset_index(drop=True)
    return table, preds, metrics


def save_model(pipeline, name: str, metadata: dict) -> Path:
    import joblib

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    # compress: the voting ensemble is ~70 trees and lands near 100 MB raw
    joblib.dump(pipeline, path, compress=3)
    meta_path = config.MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def load_model(name: str):
    import joblib

    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/train_all.py")
    return joblib.load(path)
