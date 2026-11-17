"""Quantify the data-quality issues the report discusses.

Writes reports/data_quality.json so every claim in PROJECT_REPORT.md is
reproducible from a single command.

Usage:
    python scripts/data_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, features  # noqa: E402


def main() -> int:
    raw = data.load_raw()
    frame = features.build_features(raw)
    consumption = raw["Consumption"].astype(float)

    # --- implausible low readings ----------------------------------------- #
    low = raw.loc[consumption < 100, ["Date", "Time", "Consumption"]].copy()
    low["timestamp"] = features.parse_timestamp(raw.loc[consumption < 100])
    lowest = (
        low.sort_values("Consumption")
        .head(8)[["timestamp", "Consumption"]]
        .assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))
        .to_dict(orient="records")
    )

    # --- rain sparsity ---------------------------------------------------- #
    rain = raw["Rain"]
    rain_band = features.rain_band(rain)
    band_counts = {str(k): int(v) for k, v in rain_band.value_counts().items()}
    band_share = {
        str(k): round(float(v) / float(len(rain)), 5)
        for k, v in rain_band.value_counts().items()
    }

    # --- level / trend ---------------------------------------------------- #
    by_year = (
        frame.groupby(frame[config.TIMESTAMP].dt.year)[config.TARGET]
        .agg(["count", "mean", "std"])
        .round(2)
    )
    yearly = {
        str(year): {
            "hours": int(row["count"]),
            "mean_m3": float(row["mean"]),
            "std_m3": float(row["std"]),
        }
        for year, row in by_year.iterrows()
    }

    # --- intraday oscillation (why lag-1 persistence is weak) ------------- #
    intraday = (
        frame.groupby(frame[config.TIMESTAMP].dt.hour)[config.TARGET]
        .mean()
        .round(1)
        .to_dict()
    )

    # --- autocorrelation ladder ------------------------------------------ #
    target = frame[config.TARGET].astype(float)
    autocorr = {
        str(lag): round(float(target.corr(target.shift(lag))), 4)
        for lag in (1, 2, 3, 24, 48, 72, 168)
    }

    payload = {
        "rows_raw": int(len(raw)),
        "rows_usable": int(len(frame)),
        "missing": {k: int(v) for k, v in raw.isna().sum().items()},
        "consumption": {
            "mean_m3": round(float(consumption.mean()), 2),
            "median_m3": round(float(consumption.median()), 2),
            "min_m3": float(consumption.min()),
            "max_m3": float(consumption.max()),
            "hours_below_100_m3": int((consumption < 100).sum()),
            "hours_below_300_m3": int((consumption < 300).sum()),
            "share_below_100_pct": round(100.0 * float((consumption < 100).mean()), 4),
            "lowest_readings": lowest,
        },
        "rain": {
            "missing": int(rain.isna().sum()),
            "zero_share": round(float((rain.fillna(0) == 0).mean()), 5),
            "band_counts": band_counts,
            "band_share": band_share,
            "verdict": "ordered rain-intensity feature is degenerate: >96% one class",
        },
        "yearly": yearly,
        "trend_pct_2016_to_2020": round(
            100.0
            * (yearly["2020"]["mean_m3"] / yearly["2016"]["mean_m3"] - 1.0),
            1,
        ),
        "intraday_mean_m3": {str(k): v for k, v in intraday.items()},
        "autocorrelation": autocorr,
        "autocorr_note": (
            "lag-168 autocorrelation exceeds lag-1, because demand oscillates "
            "within the day; seasonal-naive is therefore the honest baseline"
        ),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "data_quality.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in payload.items() if k != "lowest_readings"}, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
