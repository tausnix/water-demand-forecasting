# Hourly Water Demand Forecasting

One-hour-ahead forecast of aggregate water demand for a mixed residential /
commercial / industrial district, built on **45,864 hours of real operational
data** (2016–2021) from a German regional water supplier — not a synthetic series.

**Headline:** 158.3 m³ MAE on 6,880 unseen hours (**R² 0.9588**), which is a
**44% reduction in absolute error** against the seasonal-naive baseline
(282.7 m³). The model was chosen on validation data *before* the test block was
opened.

[![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B)](https://streamlit.io)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/data%20license-CC--BY--4.0-lightgrey)

---

## Problem

A utility pumping station must decide, every hour, how much water to move and
store. Better forecasts mean fewer pump start/stops, lower energy cost and less
unnecessary storage. The task is **one-hour-ahead** demand forecasting from
demand history, calendar position and *lagged* weather observations.

## Data

| | |
|---|---|
| Source | *Hourly Water Demand of a Mixed District Recorded by Supplier* |
| Publisher | Siemens AG, collected for a study at KIT |
| Zenodo record | `11045013` |
| DOI | [10.5281/zenodo.11045013](https://doi.org/10.5281/zenodo.11045013) |
| License | CC-BY-4.0 |
| Raw rows | 46,032 hourly observations |
| Usable rows | **45,864** after the 168-hour feature warm-up |
| Span | 2016-01-01 00:00 → 2021-04-01 23:00 |
| Columns | `Date, Time, Consumption, Temperature, Rain, Sun` |
| Synthetic data | **No** |

`scripts/download_data.py` fetches the CSV and verifies a pinned SHA-256, so a
swapped or edited file fails loudly instead of training silently.

**Data quality notes found during EDA:** `Rain` has 25 missing values; 48 hours
report physically implausible consumption (< 100 m³ against a 2,550 m³ median);
rain is exactly zero in 96.8% of hours, which makes the ordered rain-intensity
feature almost constant; and mean daily demand drifts upward ~21% across the
five years.

## Results

Chronological 70 / 15 / 15 split — 32,104 / 6,880 / 6,880 rows. No shuffling.

### Validation block (this is what selection used)

| Model | MAE | RMSE | R² | sMAPE |
|---|---:|---:|---:|---:|
| **voting_regressor** | **148.97** | **206.12** | **0.9595** | **8.91%** |
| xgboost | 150.06 | 207.58 | 0.9589 | 8.92% |
| bagging_extra_tree | 153.90 | 211.63 | 0.9573 | 9.26% |
| random_forest | 153.05 | 212.87 | 0.9568 | 9.21% |
| hist_gradient_boosting | 154.70 | 214.52 | 0.9561 | 9.27% |
| ridge_yj_robust | 190.40 | 256.71 | 0.9371 | 11.39% |
| naive_lag_168 | 276.01 | 402.75 | 0.8452 | 15.00% |
| naive_lag_24 | 291.24 | 427.06 | 0.8259 | 16.20% |
| naive_lag_1 | 336.09 | 454.73 | 0.8026 | 19.78% |

Also benchmarked on this block: Optuna-tuned XGBoost and a Keras-Tuner MLP
(see `reports/metrics.json`).

### Locked test — opened exactly once

| Model | Role | MAE | RMSE | R² | sMAPE |
|---|---|---:|---:|---:|---:|
| **voting_regressor** | selected | **158.26** | **220.99** | **0.9588** | **8.32%** |
| naive_lag_168 | baseline | 282.72 | 407.03 | 0.8601 | 13.58% |
| naive_lag_24 | baseline | 286.14 | 427.59 | 0.8456 | 14.45% |
| naive_lag_1 | baseline | 363.44 | 482.06 | 0.8038 | 19.58% |

### Why lag-168 is the right baseline

It is tempting to headline a 4× improvement against `naive_lag_1`. That would be
dishonest. Intraday demand oscillates, so hour-to-hour persistence is *weak*:
measured autocorrelation is 0.901 at lag 1 but **0.929 at lag 168**. LightGBM-
free comparison, same test block:

```
lag-1 persistence MAE   363.4   ← weak baseline
lag-168 persistence     282.7   ← honest baseline
voting ensemble         158.3   ← 44.0% better
```

### Cross-validation

<!--CV_TABLE-->

## Methodology

```
download+verify → causal features → baselines → ensembles → tuning → freeze → locked test
```

- **Causal features.** Demand lags 1/2/3/24/48/72/168 h; past-only rolling
  mean/std/min/max over 3/24/168 h; EWM at span 6/24. Every one is computed from
  `y.shift(1)` or stricter — never from `y`.
- **Weather is lagged.** Only `lag_1` and `lag_24`. Using the target hour's
  observation would be leakage, because a utility never has it at decision time.
- **Calendar** features are known a priori: hour/day/month with sine–cosine
  encoding, weekend flag, season, hour-period.
- **Two preprocessing branches.** Scale-sensitive models (Ridge, MLP) get
  `median impute → Yeo-Johnson → RobustScaler`. Tree models get imputation and
  categorical encoding only — no scaling, because trees do not need it and
  pretending otherwise is decoration.
- **Encoders matched to type.** OneHot for nominal (`day_of_week_name`,
  `season`, `hour_period`), Ordinal only where the ordering is physical
  (`rain_intensity`: none < light < moderate < heavy), 0/1 for binary.
- **Selection frozen before testing.** `reports/selection.json` is written first;
  `train.locked_test()` refuses to run if it is missing or if the requested model
  does not match the frozen choice.

### Four fixes over the reference implementation

This project reproduces the reference pipeline's numbers (see
`docs/PARITY.md`) but deliberately departs from it in four places:

1. **Tuning no longer contaminates the validation score.** The reference searched
   Optuna's CV over `train + validation` and then scored the winner on
   `validation`, inflating it while its rivals stayed honest. Here the search sees
   training rows only.
2. **The selected model participates in CV.** The reference explicitly excluded
   the voting ensemble from cross-validation, leaving the headline model with a
   single validation window as evidence.
3. **Every limited-history check is tested.** `tests/test_no_leakage.py` includes
   a *truncate-the-future* test: features at time *t* must be byte-identical
   whether or not later rows exist.
4. **Deployment is a first-class deliverable**, not an afterthought.

## Project structure

```
water-demand-forecasting/
├─ app.py                       Streamlit application (5 tabs)
├─ src/waterdemand/
│  ├─ config.py                 paths, seed, split, feature contract
│  ├─ data.py                   Zenodo download + SHA-256 + audit
│  ├─ features.py               causal feature engineering
│  ├─ preprocessing.py          scale-sensitive vs tree branches
│  ├─ models.py                 the 8-regressor registry + persistence baselines
│  ├─ splits.py                 chronological split
│  ├─ train.py                  stages, selection freeze, locked test
│  ├─ evaluate.py               TimeSeriesSplit CV, segments, residuals
│  ├─ tuning.py                 Optuna + Keras Tuner (train-only search)
│  └─ inference.py              one-hour-ahead from a history window
├─ scripts/
│  ├─ download_data.py          fetch + verify
│  ├─ train_all.py              end-to-end run
│  └─ make_report.py            permutation importance + figures
├─ tests/test_no_leakage.py     13 assertions
├─ models/                      voting_regressor.joblib, xgboost.joblib, metadata
└─ reports/                     metrics.json, split.json, importance.csv, figures
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-dev.txt

python scripts/download_data.py                    # fetch + verify ~1.7 MB
python -m pytest -q                                # 13 leakage/integrity tests
python scripts/train_all.py --skip-tune            # baseline → ensemble → CV → locked test
python scripts/make_report.py                      # importance + figures
streamlit run app.py
```

A full run is ~4 minutes on CPU. `--skip-tune` omits Optuna and the Keras MLP;
drop it (and `pip install tensorflow keras-tuner`) for the complete benchmark.

## The app

Five tabs, all reading pre-computed artefacts — nothing trains at request time:

| Tab | What it shows |
|---|---|
| **Forecast** | pick a target hour, get a one-hour-ahead prediction with the persistence anchors beside it |
| **Backtest** | actual vs predicted over the locked-test window, MAE by hour of day, residual distribution, under-forecast rate |
| **Models** | the full leaderboard — validation, CV and locked test side by side |
| **Drivers** | permutation importance |
| **Methodology** | split geometry, data audit, and the leakage-control table |

### Deploying to Streamlit Community Cloud

1. Push the repo (the two `.joblib` files are committed: 25.5 MB and 1.6 MB).
2. Create the app at [share.streamlit.io](https://share.streamlit.io), entry point `app.py`.
3. Keep `requirements.txt` as-is. **It deliberately excludes TensorFlow** — the
   MLP lost the benchmark and would not fit the free tier's 1 GB RAM.

Deployment gotchas this project already handles:

- **No CUDA in the pickle.** The reference builder hard-set `device="cuda"` on
  XGBoost, which crashes on a GPU-less host. `models.make_xgboost()` defaults to
  `device="cpu"` so saved boosters load anywhere.
- **Pinned versions.** An unpickle against a different scikit-learn or XGBoost
  minor version is the number-one cause of a blank Streamlit app.
- **Artifact size.** The 70-tree bagging member compresses to 25.5 MB. The
  `xgboost` alternative is 1.6 MB and is selectable in the sidebar.

## Limitations

- **Research prototype, not a deployed utility system.** No SLA, no retraining
  schedule, no monitoring.
- **Lagged weather, not forecast weather.** In production the weather inputs must
  come from a forecast feed; the current model reads the previous hour's
  observation, which is a reasonable proxy and nothing more.
- **Distribution shift.** The test block is 2020-06 → 2021-04, i.e. a pandemic
  period, and demand trends upward across the whole series. A production system
  should refit on a rolling window.
- **No prediction intervals.** Point forecasts only; the Backtest tab exposes the
  90% empirical error band as a stand-in.
- **Single series.** One district. Multi-site generalisation is untested.

## Reproducibility

Library versions move, and tree ensembles are not bit-identical across
scikit-learn or XGBoost releases. This build reproduces the reference metrics to
within **0.01 m³ MAE** and **0.0002 sMAPE** on the locked test; see
`docs/PARITY.md` for the side-by-side table and the exact cause of the residual
gap.

## License

Code: MIT. Data: CC-BY-4.0, © Siemens AG / KIT.
