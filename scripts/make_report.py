"""Pre-compute permutation importance and the report figures.

Kept out of the app's request path so the Streamlit deployment stays inside its
RAM and latency budget.

Usage:
    python scripts/make_report.py [--sample 4000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, evaluate, features, splits, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000, help="validation rows used for importance")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    frame = features.build_features(data.load_raw())
    _, validation, _, _ = splits.split_frame(frame)

    model_name = "voting_regressor"
    if not (config.MODELS_DIR / f"{model_name}.joblib").exists():
        model_name = "xgboost"
    pipeline = train.load_model(model_name)

    rng = np.random.default_rng(config.SEED)
    if len(validation) > args.sample:
        idx = rng.choice(len(validation), args.sample, replace=False)
        sample = validation.iloc[np.sort(idx)]
    else:
        sample = validation

    from sklearn.inspection import permutation_importance

    print(f"[report] permutation importance on {len(sample):,} rows, "
          f"{args.repeats} repeats, model={model_name}")
    result = permutation_importance(
        pipeline,
        sample[config.FEATURES],
        sample[config.TARGET],
        n_repeats=args.repeats,
        random_state=config.SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    importance = (
        pd.DataFrame(
            {
                "feature": config.FEATURES,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(config.REPORTS_DIR / "importance.csv", index=False)
    print(importance.head(12).to_string(index=False))

    _figures(frame, validation, pipeline, model_name)
    return 0


def _figures(frame, validation, pipeline, model_name) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. the series with split boundaries
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(frame[config.TIMESTAMP], frame[config.TARGET], linewidth=0.4, color="#0e7490")
    for ts in (
        frame[config.TIMESTAMP].iloc[int(len(frame) * 0.70)],
        frame[config.TIMESTAMP].iloc[int(len(frame) * 0.85)],
    ):
        ax.axvline(ts, color="#b45309", linestyle="--", linewidth=1)
    ax.set_title("Hourly demand, 2016–2021 (dashed = split boundaries)")
    ax.set_ylabel("m³ / hour")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "01_series_and_splits.png", dpi=140)
    plt.close(fig)

    # 2. weekly profile — explains why lag_168 beats lag_1
    fig, ax = plt.subplots(figsize=(7, 3.6))
    frame.groupby(frame[config.TIMESTAMP].dt.hour)[config.TARGET].mean().plot(ax=ax, color="#0e7490")
    ax.set_title("Mean demand by hour of day")
    ax.set_xlabel("hour")
    ax.set_ylabel("m³ / hour")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "02_intraday_profile.png", dpi=140)
    plt.close(fig)

    # 3. actual vs predicted, first two weeks of validation
    pred = pipeline.predict(validation[config.FEATURES])
    n = min(len(validation), 24 * 14)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(validation[config.TIMESTAMP].iloc[:n], validation[config.TARGET].iloc[:n],
            label="Actual", linewidth=1.4, color="#334155")
    ax.plot(validation[config.TIMESTAMP].iloc[:n], pred[:n],
            label="Predicted", linewidth=1.1, color="#0e7490")
    ax.set_title(f"Validation forecast — {model_name}")
    ax.set_ylabel("m³ / hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "03_validation_forecast.png", dpi=140)
    plt.close(fig)

    print(f"[report] figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
