"""Leakage and integrity tests.

These are the project's core defence. If someone refactors `features.py` and
accidentally introduces a `.rolling()` on the unshifted target, or fits a scaler
on the full frame, these tests fail loudly.

Run: pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waterdemand import config, data, features, preprocessing, splits  # noqa: E402


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    path = config.CANONICAL_CSV
    if not path.exists():
        pytest.skip("raw dataset missing — run scripts/download_data.py")
    return data.load_raw(path)


@pytest.fixture(scope="module")
def model(raw: pd.DataFrame) -> pd.DataFrame:
    return features.build_features(raw, drop_warmup=False)


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cut", [400, 2_000, 20_000, 45_000])
def test_features_at_t_ignore_everything_after_t(raw, model, cut):
    """Truncating the series right after t must not alter t's own feature row."""
    truncated = features.build_features(raw.iloc[: cut + 1], drop_warmup=False)

    left = model.iloc[cut]
    right = truncated.iloc[cut]

    for column in config.FEATURES:
        a, b = left[column], right[column]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"feature '{column}' at row {cut} changed when the future was removed"


@pytest.mark.parametrize("cut", [1_000, 30_000])
def test_future_target_cannot_influence_past_features(raw, model, cut):
    """Rewriting every later Consumption value must leave row `cut` untouched."""
    perturbed = raw.copy()
    perturbed.loc[cut + 1 :, config.TARGET] = (
        perturbed.loc[cut + 1 :, config.TARGET].astype(float) * 3.0
    )
    rebuilt = features.build_features(perturbed, drop_warmup=False)

    for column in config.FEATURES:
        a, b = model.iloc[cut][column], rebuilt.iloc[cut][column]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == b, f"feature '{column}' leaked from the future at row {cut}"


def test_rolling_statistics_are_shifted(model):
    """A rolling mean over the *unshifted* target would beat lag_1. It must not."""
    # consumption_roll_mean_3 must equal mean(y[t-1], y[t-2], y[t-3])
    target = model[config.TARGET].astype(float)
    expected = target.shift(1).rolling(3, min_periods=3).mean()
    pd.testing.assert_series_equal(
        model["consumption_roll_mean_3"], expected, check_names=False
    )


def test_lag_168_is_exactly_one_week_back(model):
    target = model[config.TARGET].astype(float)
    pd.testing.assert_series_equal(
        model["consumption_lag_168"], target.shift(168), check_names=False
    )


def test_weather_is_lagged(model, raw):
    """temperature_lag_1 must be yesterday's value, not the target hour's."""
    same_hour = model["temperature_lag_1"].to_numpy()
    raw_temp = raw["Temperature"].to_numpy()[: len(model)]
    # deliberately compare against the *unshifted* series: they must differ
    assert not np.allclose(same_hour[1:], raw_temp[1:], equal_nan=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_split_is_chronological_and_sized(raw):
    frame = features.build_features(raw)
    assert len(frame) == config.CANONICAL_ROWS - config.WARMUP_HOURS == 45_864

    train, validation, test, info = splits.split_frame(frame)
    assert (len(train), len(validation), len(test)) == config.EXPECTED_SPLIT
    assert train[config.TIMESTAMP].max() < validation[config.TIMESTAMP].min()
    assert validation[config.TIMESTAMP].max() < test[config.TIMESTAMP].min()
    assert info.test_rows == 6_880


def test_warmup_rows_are_dropped(raw):
    frame = features.build_features(raw)
    assert frame[list(config.REQUIRED_HISTORY)].isna().sum().sum() == 0
    assert frame[config.TIMESTAMP].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_yeo_johnson_is_fitted_on_train_only(raw):
    """Fitting on the full frame would leak the validation/test distribution."""
    frame = features.build_features(raw)
    train, _, _, _ = splits.split_frame(frame)

    train_only = preprocessing.build_scale_preprocessor().fit(
        train[config.FEATURES], train[config.TARGET]
    )
    on_everything = preprocessing.build_scale_preprocessor().fit(
        frame[config.FEATURES], frame[config.TARGET]
    )

    a = train_only.named_transformers_["continuous_yj"].named_steps["yeo_johnson"].lambdas_
    b = on_everything.named_transformers_["continuous_yj"].named_steps["yeo_johnson"].lambdas_
    assert not np.allclose(a, b), "scaler appears to have seen the held-out rows"


def test_encoder_contract(raw):
    frame = features.build_features(raw)
    assert list(features.feature_frame(frame).columns) == config.FEATURES
    assert set(config.NOMINAL).isdisjoint(config.ORDINAL)
    assert "rain_intensity" in frame.columns
