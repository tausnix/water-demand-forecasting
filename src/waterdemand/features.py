"""Causal feature engineering.

The single invariant this module must uphold, and which
``tests/test_no_leakage.py`` enforces: **the feature row for time t is a function
of observations at times <= t only.** Concretely —

* every demand lag uses ``y.shift(k)`` with ``k >= 1``;
* every rolling statistic is computed on ``y.shift(1)``, then `.rolling(window)`;
* EWMs are computed on ``y.shift(1)``;
* weather is used only at lag 1 and lag 24 (a production system would consume a
  weather *forecast*; using the target hour's observation would be leakage);
* calendar features are known a priori and are therefore safe at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #


def parse_timestamp(frame: pd.DataFrame) -> pd.Series:
    """`Date` + the starting hour of the `Time` band (" 00:00-01:00 " -> hour 0)."""
    start_hour = (
        frame["Time"]
        .astype(str)
        .str.strip()
        .str.extract(r"^(\d{1,2}):", expand=False)
        .astype(int)
    )
    return pd.to_datetime(frame["Date"], errors="raise") + pd.to_timedelta(start_hour, unit="h")


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #


def season_from_month(month: pd.Series) -> np.ndarray:
    return np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )


def hour_period(hour: pd.Series) -> np.ndarray:
    return np.select(
        [hour.between(0, 5), hour.between(6, 11), hour.between(12, 17)],
        ["night", "morning", "afternoon"],
        default="evening",
    )


def rain_band(rain: pd.Series) -> pd.Series:
    """Ordered auxiliary feature whose *ordering* is physically meaningful."""
    return pd.cut(
        rain,
        bins=[-np.inf, 0.0, 0.5, 2.0, np.inf],
        labels=["none", "light", "moderate", "heavy"],
        ordered=True,
    )


# --------------------------------------------------------------------------- #
# Main feature builder
# --------------------------------------------------------------------------- #


def build_features(frame: pd.DataFrame, *, drop_warmup: bool = True) -> pd.DataFrame:
    """Return the modelling frame: timestamp, target and every engineered feature."""
    model = frame.copy()
    model[config.TIMESTAMP] = parse_timestamp(model)
    model = (
        model.sort_values(config.TIMESTAMP)
        .drop_duplicates(config.TIMESTAMP, keep="last")
        .reset_index(drop=True)
    )

    ts = model[config.TIMESTAMP]

    # --- calendar (known at forecast time) --------------------------------- #
    model["hour"] = ts.dt.hour
    model["dow"] = ts.dt.day_of_week
    model["day_of_week_name"] = ts.dt.day_name()
    model["month"] = ts.dt.month
    model["day_of_year"] = ts.dt.day_of_year
    model["is_weekend"] = (model["dow"] >= 5).astype(int)
    model["season"] = season_from_month(model["month"])
    model["hour_period"] = hour_period(model["hour"])

    # cyclical: keep 23:00 adjacent to 00:00 and December adjacent to January
    model["hour_sin"] = np.sin(2 * np.pi * model["hour"] / 24)
    model["hour_cos"] = np.cos(2 * np.pi * model["hour"] / 24)
    model["dow_sin"] = np.sin(2 * np.pi * model["dow"] / 7)
    model["dow_cos"] = np.cos(2 * np.pi * model["dow"] / 7)
    model["month_sin"] = np.sin(2 * np.pi * (model["month"] - 1) / 12)
    model["month_cos"] = np.cos(2 * np.pi * (model["month"] - 1) / 12)

    # --- demand history: strictly past ------------------------------------- #
    target = model[config.TARGET].astype(float)
    lagged = target.shift(1)

    for lag in config.LAGS:
        model[f"consumption_lag_{lag}"] = target.shift(lag)

    for window in config.ROLL_WINDOWS:
        roll = lagged.rolling(window, min_periods=window)
        for stat in config.ROLL_STATS:
            model[f"consumption_roll_{stat}_{window}"] = getattr(roll, stat)()

    for span in config.EWM_SPANS:
        model[f"consumption_ewm_{span}"] = lagged.ewm(span=span, adjust=False).mean()

    model["consumption_prev_change"] = target.shift(1) - target.shift(2)

    # --- weather: lagged observations only --------------------------------- #
    for col in config.WEATHER_COLS:
        slug = col.lower()
        model[f"{slug}_lag_1"] = model[col].shift(1)
        model[f"{slug}_lag_24"] = model[col].shift(24)
        model[f"{slug}_roll_mean_24"] = (
            model[col].shift(1).rolling(24, min_periods=24).mean()
        )

    # rain band is derived from the *lagged* value, never the target hour
    model["rain_intensity"] = rain_band(model["rain_lag_1"])

    if drop_warmup:
        model = model.dropna(
            subset=[config.TARGET, *config.REQUIRED_HISTORY]
        ).reset_index(drop=True)

    return model


def feature_frame(model: pd.DataFrame) -> pd.DataFrame:
    """Select the model matrix in the contracted column order."""
    missing = [c for c in config.FEATURES if c not in model.columns]
    if missing:
        raise KeyError(f"feature frame is missing contracted columns: {missing}")
    return model[config.FEATURES]
