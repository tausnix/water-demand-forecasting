"""Run hyperparameter search once, then reuse it.

Tuning is the slowest stage, so it is separated from `train_all.py`. Results land
in reports/optuna_xgb.json and reports/keras_mlp.json; `train_all.py
--reuse-tuning` picks them up.

Usage:
    python scripts/tune.py                    # 30 Optuna trials + Keras Tuner
    python scripts/tune.py --trials 8 --keras-trials 3   # quick pass
    python scripts/tune.py --only optuna
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, features, splits, tuning  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30, help="Optuna trials")
    ap.add_argument("--keras-trials", type=int, default=15, help="Keras Tuner trials")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--only", choices=["optuna", "keras"], default=None)
    args = ap.parse_args()

    started = time.perf_counter()
    frame = features.build_features(data.load_raw())
    train_df, validation_df, _, _ = splits.split_frame(frame)
    print(f"tuning on {len(train_df):,} training rows; "
          f"validation ({len(validation_df):,}) is touched once, at the end")

    if args.only in (None, "optuna"):
        print(f"\n=== Optuna: {args.trials} trials, 3 expanding-window folds ===")
        params, metrics = tuning.optuna_xgb(
            train_df,
            validation_df,
            n_trials=args.trials,
            device=args.device,
        )
        print(f"\nbest CV MAE (train-only) : {json.loads((config.REPORTS_DIR / 'optuna_xgb.json').read_text())['best_value_cv_mae_train_only']:.3f}")
        print(f"validation MAE           : {metrics['mae']:.3f}")
        print(f"validation R2            : {metrics['r2']:.4f}")
        print(f"best params              : {json.dumps(params)}")

    if args.only in (None, "keras"):
        print(f"\n=== Keras Tuner: {args.keras_trials} trials, {args.epochs} epochs ===")
        try:
            metrics = tuning.keras_mlp(
                train_df,
                validation_df,
                max_trials=args.keras_trials,
                epochs=args.epochs,
            )
            print(f"validation MAE : {metrics['mae']:.3f}")
            print(f"validation R2  : {metrics['r2']:.4f}")
        except ImportError as exc:
            print(f"[keras] unavailable: {exc}")
            return 1

    print(f"\nDONE in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
