"""One-hour-ahead inference from a window of recent history.

The subtle part: to forecast time T+1 we append a *blank* row at T+1 and let the
causal feature builder fill it from observations up to T. Because every lag is a
`shift(k>=1)`, that blank row's features are well defined even though its
Consumption is unknown. Its unshifted weather is never read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


class InsufficientHistory(ValueError):
    """Raised when fewer than the required 168 hours of history are available."""


def _format_date(ts: pd.Timestamp) -> str:
    # the published file uses unpadded m/d/Y ("1/1/2016")
    return f"{ts.month}/{ts.day}/{ts.year}"


def _format_time(ts: pd.Timestamp) -> str:
    # only the leading hour is parsed by parse_timestamp
    return f" {ts.hour:02d}:00 "


def build_forecast_frame(history: pd.DataFrame, next_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Append a placeholder row at `next_timestamp` and engineer causal features."""
    required = ["Date", "Time", "Consumption", "Temperature", "Rain", "Sun"]
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise KeyError(f"history is missing columns: {missing}")

    ordered = history.copy()
    ordered["timestamp"] = features.parse_timestamp(ordered)
    ordered = ordered.sort_values("timestamp").reset_index(drop=True)

    if len(ordered) < config.WARMUP_HOURS:
        raise InsufficientHistory(
            f"need at least {config.WARMUP_HOURS} hours of history, have {len(ordered)}"
        )

    placeholder = {
        "Date": _format_date(next_timestamp),
        "Time": _format_time(next_timestamp),
        "Consumption": np.nan,
        "Temperature": np.nan,
        "Rain": np.nan,
        "Sun": np.nan,
    }
    combined = pd.concat([ordered[required], pd.DataFrame([placeholder])], ignore_index=True)

    engineered = features.build_features(combined, drop_warmup=False)
    return engineered.tail(1).reset_index(drop=True)


def forecast_next_hour(history: pd.DataFrame, pipeline, next_timestamp=None) -> dict:
    """Return the point forecast plus the persistence anchors for context."""
    ordered = history.copy()
    ordered["timestamp"] = features.parse_timestamp(ordered)
    latest = ordered["timestamp"].max()
    next_ts = next_timestamp or (latest + pd.Timedelta(hours=1))

    row = build_forecast_frame(ordered, next_ts)
    x = row[config.FEATURES]

    prediction = float(pipeline.predict(x)[0])

    anchors = {}
    for lag in (1, 24, 168):
        column = f"consumption_lag_{lag}"
        value = row[column].iloc[0]
        anchors[f"lag_{lag}"] = None if pd.isna(value) else float(value)

    return {
        "timestamp": next_ts,
        "prediction": prediction,
        "persistence": anchors,
        "last_observed": float(ordered.sort_values("timestamp")[config.TARGET].iloc[-1]),
    }
