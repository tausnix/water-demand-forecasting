"""Single source of truth for paths, seeds, split geometry and the feature contract.

Every constant the pipeline depends on lives here so the notebook, the training
script, the tests and the Streamlit app can never silently disagree.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CANONICAL_CSV = DATA_RAW / "waterconsumption_noNaN_fixedTimechange.csv"
FEATURES_CSV = DATA_INTERIM / "features.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
SPLIT_JSON = REPORTS_DIR / "split.json"
SELECTION_JSON = REPORTS_DIR / "selection.json"

# --------------------------------------------------------------------------- #
# Dataset provenance — pinned so a swapped/edited CSV fails loudly
# --------------------------------------------------------------------------- #
ZENODO_RECORD = "11045013"
ZENODO_DOI = "10.5281/zenodo.11045013"
ZENODO_URL = (
    "https://zenodo.org/api/records/11045013/files/"
    "waterconsumption_noNaN_fixedTimechange.csv/content"
)
CANONICAL_SHA256 = "3acf63ff46af582c00748a0d46fc4f10b1038a4eaf542a9cb882275d480b093a"
CANONICAL_ROWS = 46_032

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Chronological split: 70 / 15 / 15.  Test is locked until selection is frozen.
# 45_864 usable rows -> 32_104 / 6_880 / 6_880
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
EXPECTED_SPLIT = (32_104, 6_880, 6_880)

# --------------------------------------------------------------------------- #
# Feature engineering contract
# --------------------------------------------------------------------------- #
TARGET = "Consumption"
TIMESTAMP = "timestamp"

WARMUP_HOURS = 168  # longest lag; rows 0..167 carry no complete history

LAGS = (1, 2, 3, 24, 48, 72, 168)
ROLL_WINDOWS = (3, 24, 168)
ROLL_STATS = ("mean", "std", "min", "max")
EWM_SPANS = (6, 24)
WEATHER_COLS = ("Temperature", "Rain", "Sun")

# The reference implementation's warm-up rule: drop rows lacking these.
REQUIRED_HISTORY = ("consumption_lag_168", "consumption_roll_mean_168", "consumption_roll_std_168")

# --------------------------------------------------------------------------- #
# Encoder assignment — recorded explicitly so the README cannot drift from code
# --------------------------------------------------------------------------- #
NOMINAL = ["day_of_week_name", "season", "hour_period"]
ORDINAL = ["rain_intensity"]
RAIN_BANDS = [["none", "light", "moderate", "heavy"]]
BINARY = ["is_weekend"]
CYCLICAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

CONTINUOUS = [
    *(f"consumption_lag_{lag}" for lag in LAGS),
    *(
        f"consumption_roll_{stat}_{win}"
        for win in ROLL_WINDOWS
        for stat in ROLL_STATS
    ),
    *(f"consumption_ewm_{span}" for span in EWM_SPANS),
    "consumption_prev_change",
    "temperature_lag_1",
    "temperature_lag_24",
    "temperature_roll_mean_24",
    "rain_lag_1",
    "rain_lag_24",
    "rain_roll_mean_24",
    "sun_lag_1",
    "sun_lag_24",
    "sun_roll_mean_24",
]

# Order is part of the contract: pickled preprocessors address columns by index.
FEATURES = CONTINUOUS + CYCLICAL + BINARY + NOMINAL + ORDINAL

assert len(FEATURES) == len(set(FEATURES)), "duplicate feature name in contract"
