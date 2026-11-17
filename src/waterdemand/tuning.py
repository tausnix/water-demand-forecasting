"""Hyperparameter search — deliberately narrower than the reference implementation.

**The fix:** the reference runs Optuna's CV over `train + validation`, then scores
the winning parameters on `validation`. That makes the tuned model's validation
MAE optimistically biased while its competitors' scores stay honest, so the
selection table compares apples to oranges. Here the search sees **training rows
only**; validation is touched once, after the trial loop closes.

Same rule for the MLP: early stopping monitors a slice carved out of *training*
data, not the validation block.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from . import config, models
from .metrics import regression_metrics


def optuna_xgb(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    n_trials: int = 30,
    n_splits: int = 3,
    timeout: int | None = 900,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Search on training folds only, then report one honest validation score."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 180, 500, step=80),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        scores = []
        for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(train_df):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]  # <-- training rows only
            pipeline = models.build_pipeline("xgboost", params)
            pipeline.fit(tr[config.FEATURES], tr[config.TARGET])
            pred = pipeline.predict(va[config.FEATURES])
            scores.append(regression_metrics(va[config.TARGET], pred)["mae"])
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # one clean look at validation with the frozen parameters
    pipeline = models.build_pipeline("tuned_xgboost", study.best_params)
    pipeline.fit(train_df[config.FEATURES], train_df[config.TARGET])
    pred = pipeline.predict(validation_df[config.FEATURES])
    metrics = regression_metrics(validation_df[config.TARGET], pred)

    payload = {
        "best_value_cv_mae_train_only": study.best_value,
        "best_params": study.best_params,
        "trials_completed": len(study.trials),
        "search_data": "training rows only",
        "validation": metrics,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "optuna_xgb.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"number": t.number, "value": t.value, "state": str(t.state), **t.params}
            for t in study.trials
        ]
    ).to_csv(config.REPORTS_DIR / "optuna_trials.csv", index=False)

    return study.best_params, metrics


def keras_mlp(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_trials: int = 15,
    epochs: int = 25,
    batch_size: int = 512,
) -> dict:
    """Keras Tuner MLP. Training-only: it is benchmarked, never served."""
    import keras_tuner as kt
    import tensorflow as tf

    from .preprocessing import build_scale_preprocessor

    pre = build_scale_preprocessor().fit(train_df[config.FEATURES], train_df[config.TARGET])
    x_train = np.asarray(pre.transform(train_df[config.FEATURES]), dtype="float32")
    x_val = np.asarray(pre.transform(validation_df[config.FEATURES]), dtype="float32")

    y_train = train_df[config.TARGET].to_numpy("float32")
    y_val = validation_df[config.TARGET].to_numpy("float32")
    mean, std = float(y_train.mean()), float(y_train.std() or 1.0)
    y_train_n = (y_train - mean) / std
    y_val_n = (y_val - mean) / std

    # early stopping monitors an internal slice of TRAINING data, not validation
    split = int(len(x_train) * 0.9)
    x_fit, y_fit = x_train[:split], y_train_n[:split]
    x_stop, y_stop = x_train[split:], y_train_n[split:]

    def build(hp: "kt.HyperParameters") -> "tf.keras.Model":
        inputs = tf.keras.Input(shape=(x_train.shape[1],))
        x = inputs
        for i in range(hp.Int("hidden_layers", 1, 3)):
            units = hp.Choice(f"units_{i}", [64, 128, 256])
            x = tf.keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
            if hp.Boolean(f"batchnorm_{i}"):
                x = tf.keras.layers.BatchNormalization()(x)
            dropout = hp.Choice(f"dropout_{i}", [0.0, 0.1, 0.2, 0.3])
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        net = tf.keras.Model(inputs, outputs)
        net.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float("learning_rate", 1e-4, 3e-3, sampling="log")
            ),
            loss="mse",
            metrics=["mae"],
        )
        return net

    tuner = kt.RandomSearch(
        build,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=True,
        directory=str(config.REPORTS_DIR / "keras_tuner"),
        project_name="mlp",
        seed=config.SEED,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    tuner.search(
        x_fit,
        y_fit,
        validation_data=(x_stop, y_stop),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    best = tuner.get_best_hyperparameters(1)[0]
    net = tuner.hypermodel.build(best)
    net.fit(
        x_fit,
        y_fit,
        validation_data=(x_stop, y_stop),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    pred = net.predict(x_val, batch_size=1024, verbose=0).reshape(-1) * std + mean
    metrics = regression_metrics(y_val, pred)

    payload = {
        "best_hyperparameters": best.values,
        "validation_metrics": metrics,
        "target_mean": mean,
        "target_std": std,
    }
    (config.REPORTS_DIR / "keras_mlp.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return metrics
