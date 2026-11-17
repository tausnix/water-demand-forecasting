"""End-to-end training: baselines -> ensembles -> CV -> tuning -> locked test.

Usage:
    python scripts/train_all.py                  # full run
    python scripts/train_all.py --skip-tune      # skip Optuna / Keras Tuner
    python scripts/train_all.py --skip-cv --skip-tune --quick   # smoke run

Artifacts written:
    reports/metrics.json, reports/split.json, reports/selection.json
    reports/locked_test_predictions.csv
    models/<name>.joblib, models/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, evaluate, features, models, splits, train  # noqa: E402


def _load_cached_tuning() -> tuple[dict, dict]:
    """Read tuning artefacts written by scripts/tune.py.

    Keeps the pipeline idempotent: tuning is the expensive, slow-moving part, so
    it is run once and reused while the rest of the pipeline is iterated on.
    """
    rows: dict[str, dict] = {}
    params: dict[str, dict] = {}

    xgb_path = config.REPORTS_DIR / "optuna_xgb.json"
    if xgb_path.exists():
        payload = json.loads(xgb_path.read_text(encoding="utf-8"))
        if "validation" in payload:
            rows["tuned_xgboost"] = {
                "model": "tuned_xgboost",
                "stage": "tuned",
                **payload["validation"],
            }
            params["tuned_xgboost"] = payload["best_params"]
    else:
        print(f"  [warn] {xgb_path.name} missing")

    mlp_path = config.REPORTS_DIR / "keras_mlp.json"
    if mlp_path.exists():
        payload = json.loads(mlp_path.read_text(encoding="utf-8"))
        if "validation_metrics" in payload:
            rows["keras_mlp"] = {
                "model": "keras_mlp",
                "stage": "neural",
                **payload["validation_metrics"],
            }
    else:
        print(f"  [warn] {mlp_path.name} missing")

    return rows, params


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-cv", action="store_true", help="skip TimeSeriesSplit")
    ap.add_argument("--skip-tune", action="store_true", help="skip Optuna and Keras Tuner")
    ap.add_argument(
        "--reuse-tuning",
        action="store_true",
        help="reuse reports/optuna_xgb.json and keras_mlp.json instead of re-running tuning",
    )
    ap.add_argument("--quick", action="store_true", help="fewer CV folds / Optuna trials")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    print("=== 1. DATA ===")
    raw = data.load_raw()
    report = data.audit(raw, config.CANONICAL_CSV)
    data.save_audit(report)
    print(f"raw rows: {len(raw):,}  missing: {report['missing']}")

    frame = features.build_features(raw)
    print(f"usable rows after {config.WARMUP_HOURS}h warm-up: {len(frame):,}")
    if len(frame) != 45_864:
        raise RuntimeError(f"expected 45,864 usable rows, got {len(frame)}")
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.FEATURES_CSV, index=False)

    train_df, validation_df, test_df, split_info = splits.split_frame(frame)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLIT_JSON.write_text(
        json.dumps(splits.as_dict(split_info), indent=2) + "\n", encoding="utf-8"
    )
    print(f"split: {split_info.train_rows:,} / {split_info.validation_rows:,} / {split_info.test_rows:,}")

    print("\n=== 2. BASELINES (validation) ===")
    baseline_table, baseline_preds = train.baseline_stage(train_df, validation_df)
    print(baseline_table[["model", "mae", "rmse", "r2", "smape_pct"]].to_string(index=False))

    print("\n=== 3. ENSEMBLES (validation) ===")
    ensemble_table, ensemble_preds = train.ensemble_stage(train_df, validation_df)
    print(ensemble_table[["model", "mae", "rmse", "r2", "smape_pct"]].to_string(index=False))

    valid_table = pd.concat([baseline_table, ensemble_table], ignore_index=True)
    tuned_rows: dict[str, dict] = {}
    tuned_params: dict[str, dict] = {}

    if not args.skip_cv:
        print("\n=== 4. TIME-SERIES CV (train+val, refit per fold) ===")
        cv_names = ["xgboost", "bagging_extra_tree", "hist_gradient_boosting", "voting_regressor"]
        folds, cv_summary = evaluate.time_series_cv(
            pd.concat([train_df, validation_df], ignore_index=True),
            cv_names,
            n_splits=3 if args.quick else 5,
            params=None,
        )
        folds.to_csv(config.REPORTS_DIR / "cv_folds.csv", index=False)
        cv_summary.to_csv(config.REPORTS_DIR / "cv_summary.csv", index=False)
        print(cv_summary.to_string(index=False))
    else:
        cv_summary = pd.DataFrame()

    if args.reuse_tuning:
        print("\n=== 5. TUNING (reusing reports/optuna_xgb.json + keras_mlp.json) ===")
        tuned_rows, tuned_params = _load_cached_tuning()
        for name, row in tuned_rows.items():
            print(f"  {name}: validation MAE={row['mae']:.3f}")
    elif not args.skip_tune:
        print("\n=== 5. TUNING (scored on validation only) ===")
        from waterdemand import tuning

        xgb_params, xgb_val = tuning.optuna_xgb(
            train_df, validation_df, n_trials=8 if args.quick else 30, device=args.device
        )
        tuned_rows["tuned_xgboost"] = {"model": "tuned_xgboost", "stage": "tuned", **xgb_val}
        tuned_params["tuned_xgboost"] = xgb_params
        print(f"tuned xgboost validation MAE={xgb_val['mae']:.3f}")

        try:
            mlp_val = tuning.keras_mlp(train_df, validation_df)
            tuned_rows["keras_mlp"] = {"model": "keras_mlp", "stage": "neural", **mlp_val}
            print(f"keras mlp validation MAE={mlp_val['mae']:.3f}")
        except Exception as exc:  # TF absent is not a reason to lose the whole run
            print(f"[tuning] Keras MLP skipped: {type(exc).__name__}: {exc}")

    print("\n=== 6. SELECT (validation only, test still unread) ===")
    selection = train.select(valid_table, list(tuned_rows.values()) or None)
    print(json.dumps(selection, indent=2))

    print("\n=== 7. LOCKED TEST (single read) ===")
    locked_table, locked_preds, locked_metrics = train.locked_test(
        train_df,
        validation_df,
        test_df,
        selection["selected_before_test"],
        tuned_params.get(selection["selected_before_test"]),
    )
    print(locked_table.to_string(index=False))

    naive_mae = float(locked_table.loc[locked_table["model"] == "naive_lag_168", "mae"].iloc[0])
    skill = 1.0 - locked_metrics["mae"] / naive_mae
    print(f"\nskill vs seasonal-naive (lag 168): {skill:.1%} MAE reduction")

    print("\n=== 8. SAVE ARTIFACTS ===")
    selected_name = selection["selected_before_test"]
    development = pd.concat([train_df, validation_df], ignore_index=True)
    pipeline = models.build_pipeline(selected_name, tuned_params.get(selected_name))
    pipeline.fit(development[config.FEATURES], development[config.TARGET])
    model_path = train.save_model(
        pipeline,
        selected_name,
        {
            "selected_model": selected_name,
            "features": config.FEATURES,
            "seed": config.SEED,
            "trained_on": {
                "rows": int(len(development)),
                "start": str(development[config.TIMESTAMP].min()),
                "end": str(development[config.TIMESTAMP].max()),
            },
            "locked_test": locked_metrics,
            "skill_vs_naive_lag_168": skill,
        },
    )
    print(f"model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Lightweight alternative for constrained hosting: ~6 MB vs ~26 MB, and its
    # validation MAE is within the noise band of the winner.
    if selected_name != "xgboost":
        light = models.build_pipeline("xgboost")
        light.fit(development[config.FEATURES], development[config.TARGET])
        light_path = train.save_model(
            light,
            "xgboost",
            {
                "selected_model": "xgboost",
                "role": "lightweight alternative",
                "features": config.FEATURES,
                "seed": config.SEED,
            },
        )
        print(f"light model: {light_path} ({light_path.stat().st_size / 1e6:.1f} MB)")

    payload = {
        "dataset": {
            "zenodo_record": config.ZENODO_RECORD,
            "doi": config.ZENODO_DOI,
            "synthetic": False,
            "raw_rows": report["rows"],
            "usable_rows": int(len(frame)),
            "sha256": report["sha256"],
        },
        "split": splits.as_dict(split_info),
        "feature_count": len(config.FEATURES),
        "validation": valid_table.to_dict(orient="records"),
        "tuned_validation": list(tuned_rows.values()),
        "tuned_params": tuned_params,
        "cv": cv_summary.to_dict(orient="records"),
        "selection": selection,
        "locked_test": locked_table.to_dict(orient="records"),
        "skill_vs_naive_lag_168": skill,
        "residuals": evaluate.residual_summary(
            test_df[config.TARGET], locked_preds[selected_name].to_numpy()
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    config.METRICS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    locked_table.to_csv(config.REPORTS_DIR / "locked_test_metrics.csv", index=False)
    locked_preds.to_csv(config.REPORTS_DIR / "locked_test_predictions.csv", index=False)
    print(f"metrics: {config.METRICS_JSON}")

    print(f"\nDONE in {payload['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
