# Parity with the reference implementation

The project brief specified exact figures (R² 0.9588, MAE 158.25, naive 282.72).
This document records how closely this build reproduces them, and why any gap
exists.

Reference: [`NikunjGit-config-Js/Water-Demanding-Forecast`](https://github.com/NikunjGit-config-Js/Water-Demanding-Forecast)
`artifacts/results/` — produced on Linux with TensorFlow 2.20 and an earlier
scikit-learn / XGBoost pair.

## Split — byte-identical

| Field | Reference | This build |
|---|---|---|
| Usable rows | 45,864 | 45,864 |
| Train / val / test | 32,104 / 6,880 / 6,880 | 32,104 / 6,880 / 6,880 |
| Train ends | 2019-09-06T15:00:00 | 2019-09-06T15:00:00 |
| Validation | 2019-09-06T16:00 → 2020-06-19T07:00 | identical |
| Test | 2020-06-19T08:00 → 2021-04-01T23:00 | identical |

## Validation block

| Model | Reference MAE | This build | Δ |
|---|---:|---:|---:|
| naive_lag_1 | 336.087936 | 336.087936 | 0.000 |
| naive_lag_24 | 291.239826 | 291.239826 | 0.000 |
| naive_lag_168 | 276.005814 | 276.005814 | 0.000 |
| ridge_yj_robust | 190.398474 | 190.398474 | 0.000 |
| random_forest | 153.045601 | 153.045601 | 0.000 |
| bagging_extra_tree | 153.896593 | 153.896593 | 0.000 |
| hist_gradient_boosting | 154.959508 | 154.702296 | 0.257 |
| xgboost | 150.373161 | 150.061354 | 0.312 |
| voting_regressor | 149.291218 | 148.967844 | 0.323 |

## Locked test

| Metric | Reference | This build | Δ |
|---|---:|---:|---:|
| MAE | 158.251144 | **158.261093** | 0.010 |
| RMSE | 220.991933 | **220.986683** | 0.005 |
| R² | 0.9587606 | **0.9587628** | 0.0000022 |
| sMAPE | 8.321224% | **8.323460%** | 0.0022 |
| naive_lag_168 MAE | 282.719331 | 282.719331 | 0.000 |

**Verdict: reproduced.** Every deterministic estimator (naive baselines, Ridge,
RandomForest, Bagging) matches to six decimal places. The three that drift are
the ones whose implementations changed between library releases.

## Why `hist_gradient_boosting` and `xgboost` differ

Both are histogram-based boosters whose binning, tie-breaking and default
regularisation have changed since the reference run:

- **HistGradientBoosting** — scikit-learn 1.7 changed the default
  `categorical_features` handling and the early-stopping interaction.
- **XGBoost** — this build uses 3.4.1; the reference predates 3.0, which changed
  the default `tree_method` behaviour and GPU/CPU split rounding.

The voting ensemble inherits both drifts, which is why its validation MAE moves
by 0.32 m³ while the locked-test figure lands within **0.01 m³** of the
reference. Seed (`42`) and hyperparameters are identical; nothing else was
tuned to chase the number.

## Deliberate departures

These change results on purpose and are documented rather than hidden:

| Change | Reference | This build | Effect |
|---|---|---|---|
| Optuna search data | `train + val` | `train` only | Tuned model's validation MAE is no longer inflated |
| Keras early stopping | monitors `val_loss` on validation | internal 10% slice of train | Same |
| Voting in CV | excluded | included | Headline model gains CV evidence |
| XGBoost device | `cuda`, hard-coded | `cpu` default | Booster loads on GPU-less hosts |
| Dataset in repo | absent (local zip path) | downloaded + SHA-256 verified | Anyone can reproduce |

The reproducing configuration is the default: `python scripts/train_all.py`.
